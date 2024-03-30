import torch
import numpy as np
import torchvision
from torchvision.transforms import ToTensor
from typing import List

import os
import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from inference.models import YOLOWorld
import requests
from PIL import Image, ImageFilter
from io import BytesIO
import numpy as np
from huggingface_hub import hf_hub_download

GPU_EFFICIENT_SAM_CHECKPOINT = "efficient_sam_s_gpu.jit"
CPU_EFFICIENT_SAM_CHECKPOINT = "efficient_sam_s_cpu.jit"


def load(device: torch.device) -> torch.jit.ScriptModule:
    if device.type == "cuda":
        model = torch.jit.load(
            hf_hub_download(repo_id="SkalskiP/YOLO-World", filename=GPU_EFFICIENT_SAM_CHECKPOINT, repo_type="space")
        )
    else:
        model = torch.jit.load(
            hf_hub_download(repo_id="SkalskiP/YOLO-World", filename=CPU_EFFICIENT_SAM_CHECKPOINT, repo_type="space")
        )
    model.eval()
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFFICIENT_SAM_MODEL = load(device=DEVICE)
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def inference_with_box(
        image: np.ndarray,
        box: np.ndarray,
        model: torch.jit.ScriptModule,
        device: torch.device
) -> np.ndarray:
    bbox = torch.reshape(torch.tensor(box), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        bbox.to(device),
        bbox_labels.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou


def inference_with_boxes(
        image: np.ndarray,
        xyxy: np.ndarray,
        model: torch.jit.ScriptModule,
        device: torch.device
) -> np.ndarray:
    masks = []
    for [x_min, y_min, x_max, y_max] in xyxy:
        box = np.array([[x_min, y_min], [x_max, y_max]])
        mask = inference_with_box(image, box, model, device)
        masks.append(mask)
    return np.array(masks)


def get_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]


def annotate_image(
        input_image: np.ndarray,
        detections: sv.Detections,
        categories: List[str],
        negative_detections: sv.Detections = None,
        negative_categories: List[str] = [],
        with_confidence: bool = False,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    labels += [
        (
            f"{negative_categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{negative_categories[class_id]}"
        )
        for class_id, confidence in
        zip(negative_detections.class_id, negative_detections.confidence)
    ] if negative_detections else []

    combined = sv.Detections.merge([detections, negative_detections]) if negative_detections else detections
    output_image = MASK_ANNOTATOR.annotate(input_image, combined)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, combined)
    output_image = LABEL_ANNOTATOR.annotate(output_image, combined, labels=labels)
    return output_image


def process_image(
        image: Image.Image,
        categories: str,
        negative_categories: str = None,  # Optional parameter for negative categories
        confidence_threshold: float = 0.01,
        iou_threshold: float = 0.1,
        with_segmentation: bool = True,
        with_confidence: bool = False,
        with_class_agnostic_nms: bool = False,
) -> tuple:
    input_image = np.array(image)
    positive_categories = process_categories(categories)
    negative_categories = process_categories(negative_categories) if negative_categories else []

    # Initialize aggregate masks
    positive_mask_aggregate = np.zeros(input_image.shape[:2], dtype=np.uint8)
    if negative_categories:
        negative_mask_aggregate = np.zeros(input_image.shape[:2], dtype=np.uint8)

    # Handle Positive Categories
    YOLO_WORLD_MODEL.set_classes(positive_categories)
    positive_results = YOLO_WORLD_MODEL.infer(input_image, confidence=confidence_threshold)
    positive_detections = sv.Detections.from_inference(positive_results).with_nms(
        class_agnostic=with_class_agnostic_nms, threshold=iou_threshold)

    if with_segmentation:
        positive_masks = inference_with_boxes(
            image=input_image,
            xyxy=positive_detections.xyxy,
            model=EFFICIENT_SAM_MODEL,
            device=DEVICE
        )
        for mask in positive_masks:
            positive_mask_aggregate[mask == 1] = 255

    combined_detections = positive_detections

    # Handle Negative Categories
    negative_detections = None
    if negative_categories:
        YOLO_WORLD_MODEL.set_classes(negative_categories)
        negative_results = YOLO_WORLD_MODEL.infer(input_image, confidence=confidence_threshold)
        negative_detections = sv.Detections.from_inference(negative_results).with_nms(
            class_agnostic=with_class_agnostic_nms, threshold=iou_threshold)

        negative_masks = inference_with_boxes(
            image=input_image,
            xyxy=negative_detections.xyxy,
            model=EFFICIENT_SAM_MODEL,
            device=DEVICE
        )
        for mask in negative_masks:
            negative_mask_aggregate[mask == 1] = 255

        negative_mask_aggregate_image = Image.fromarray(negative_mask_aggregate).convert("L")
        negative_mask_aggregate_image = adjust_image_mask(negative_mask_aggregate_image, 16)
        negative_mask_aggregate_image = negative_mask_aggregate_image.filter(ImageFilter.GaussianBlur(radius=16))
        negative_mask_aggregate_image = negative_mask_aggregate_image.convert("L")
        negative_mask_aggregate = np.array(negative_mask_aggregate_image)

        # Subtract the negative mask from the positive mask
        positive_mask_aggregate[negative_mask_aggregate != 0] = 0
        negative_detections.class_id

        # Combine positive and negative detections
        combined_detections = sv.Detections.merge([positive_detections, negative_detections])

    # Convert the positive aggregate mask to a PIL image
    mask_image = Image.fromarray(positive_mask_aggregate)

    # Annotate Image
    output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=positive_detections,
        categories=positive_categories,
        negative_detections=negative_detections,
        negative_categories=negative_categories,
        with_confidence=with_confidence
    )
    annotated_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

    # Return the tuple of the annotated image and the mask image
    return (annotated_image, mask_image)


def adjust_image_mask(image_mask: Image.Image, shift_pixels: int) -> Image.Image:
    if shift_pixels == 0:
        return image_mask

    mask_array = np.array(image_mask.convert("L"))

    kernel = np.ones((abs(shift_pixels) * 2 + 1, abs(shift_pixels) * 2 + 1), np.uint8)

    if shift_pixels > 0:
        adjusted_mask_array = cv2.dilate(mask_array, kernel, iterations=1)
    else:
        adjusted_mask_array = cv2.erode(mask_array, kernel, iterations=1)

    adjusted_image_mask = Image.fromarray(adjusted_mask_array)

    return adjusted_image_mask
