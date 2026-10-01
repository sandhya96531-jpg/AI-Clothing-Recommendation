from pathlib import Path
from shutil import copy2

from PIL import Image, ImageDraw, ImageFilter

import numpy as np
import cv2

from gradio_client import Client, handle_file


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "storage" / "temp" / "vton_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IDM-VTON CLIENT
# ============================================================

def get_vton_client():
    return Client("yisol/IDM-VTON")


# ============================================================
# HELPER: GET LANDMARK POINT
# ============================================================

def _safe_point(landmark, width, height):
    x = int(max(0, min(width - 1, landmark.x * width)))
    y = int(max(0, min(height - 1, landmark.y * height)))

    visibility = getattr(landmark, "visibility", 1.0)

    return x, y, visibility


# ============================================================
# BUILD BETTER LOWER BODY MASK
#
# This mask focuses mainly on:
# - pelvis
# - left thigh
# - right thigh
# - left leg
# - right leg
#
# It avoids creating one huge rectangle around the whole
# lower body, which was causing the bag/arms/artifacts problem.
# ============================================================

def build_lower_body_mask(person_image):

    try:
        import mediapipe as mp
    except Exception as error:
        raise RuntimeError(
            "MediaPipe is required for lower-body virtual try-on."
        ) from error

    person_path = Path(person_image)

    image = cv2.imread(str(person_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read person image:\n{person_path}"
        )

    height, width = image.shape[:2]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:

        result = pose.process(rgb_image)

    if not result.pose_landmarks:
        raise RuntimeError(
            "Could not detect the person's pose. "
            "Please use a clear full-body photo."
        )

    landmarks = result.pose_landmarks.landmark

    # MediaPipe pose landmark indexes
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_HIP = 23
    RIGHT_HIP = 24

    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # --------------------------------------------------------
    # Read landmarks
    # --------------------------------------------------------

    ls = _safe_point(
        landmarks[LEFT_SHOULDER],
        width,
        height
    )

    rs = _safe_point(
        landmarks[RIGHT_SHOULDER],
        width,
        height
    )

    lh = _safe_point(
        landmarks[LEFT_HIP],
        width,
        height
    )

    rh = _safe_point(
        landmarks[RIGHT_HIP],
        width,
        height
    )

    lk = _safe_point(
        landmarks[LEFT_KNEE],
        width,
        height
    )

    rk = _safe_point(
        landmarks[RIGHT_KNEE],
        width,
        height
    )

    la = _safe_point(
        landmarks[LEFT_ANKLE],
        width,
        height
    )

    ra = _safe_point(
        landmarks[RIGHT_ANKLE],
        width,
        height
    )

    # Check important landmarks
    required_points = [
        lh, rh,
        lk, rk,
        la, ra
    ]

    for point in required_points:
        if point[2] < 0.35:
            raise RuntimeError(
                "Lower-body landmarks are not clear enough. "
                "Please use a full-body photo with both legs visible."
            )

    # --------------------------------------------------------
    # Extract x/y only
    # --------------------------------------------------------

    left_hip = np.array([lh[0], lh[1]], dtype=np.float32)
    right_hip = np.array([rh[0], rh[1]], dtype=np.float32)

    left_knee = np.array([lk[0], lk[1]], dtype=np.float32)
    right_knee = np.array([rk[0], rk[1]], dtype=np.float32)

    left_ankle = np.array([la[0], la[1]], dtype=np.float32)
    right_ankle = np.array([ra[0], ra[1]], dtype=np.float32)

    # --------------------------------------------------------
    # Body measurements
    # --------------------------------------------------------

    hip_distance = float(
        np.linalg.norm(right_hip - left_hip)
    )

    if hip_distance < 20:
        raise RuntimeError(
            "Could not determine the hip width properly."
        )

    shoulder_y = min(ls[1], rs[1])

    hip_y = min(lh[1], rh[1])

    knee_y = max(lk[1], rk[1])

    ankle_y = max(la[1], ra[1])

    # --------------------------------------------------------
    # Create empty mask
    # --------------------------------------------------------

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # ========================================================
    # 1. PELVIS / WAIST AREA
    #
    # Keep it relatively narrow so that arms/bag are not
    # included.
    # ========================================================

    hip_center = (
        (left_hip + right_hip) / 2.0
    )

    pelvis_top_y = int(
        hip_center[1] - hip_distance * 0.18
    )

    pelvis_bottom_y = int(
        hip_center[1] + hip_distance * 0.55
    )

    pelvis_left_x = int(
        hip_center[0] - hip_distance * 0.62
    )

    pelvis_right_x = int(
        hip_center[0] + hip_distance * 0.62
    )

    pelvis_left_x = max(
        0,
        pelvis_left_x
    )

    pelvis_right_x = min(
        width - 1,
        pelvis_right_x
    )

    pelvis_top_y = max(
        0,
        pelvis_top_y
    )

    pelvis_bottom_y = min(
        height - 1,
        pelvis_bottom_y
    )

    cv2.rectangle(
        mask,
        (
            pelvis_left_x,
            pelvis_top_y
        ),
        (
            pelvis_right_x,
            pelvis_bottom_y
        ),
        255,
        -1
    )

    # ========================================================
    # 2. FUNCTION TO CREATE A NARROW LEG CORRIDOR
    # ========================================================

    def create_leg_corridor(
        hip_point,
        knee_point,
        ankle_point,
        hip_width,
        knee_width,
        ankle_width
    ):

        points = [
            hip_point,
            knee_point,
            ankle_point
        ]

        widths = [
            hip_width,
            knee_width,
            ankle_width
        ]

        left_side = []
        right_side = []

        for index, point in enumerate(points):

            if index == 0:
                direction = points[1] - points[0]

            elif index == len(points) - 1:
                direction = points[-1] - points[-2]

            else:
                direction = points[index + 1] - points[index - 1]

            length = np.linalg.norm(direction)

            if length < 1:
                direction = np.array(
                    [0.0, 1.0],
                    dtype=np.float32
                )
                length = 1

            direction = direction / length

            # perpendicular vector
            perpendicular = np.array(
                [-direction[1], direction[0]],
                dtype=np.float32
            )

            half_width = widths[index] / 2.0

            left_side.append(
                point + perpendicular * half_width
            )

            right_side.append(
                point - perpendicular * half_width
            )

        polygon_points = (
            left_side +
            right_side[::-1]
        )

        polygon_points = np.array(
            polygon_points,
            dtype=np.int32
        )

        return polygon_points

    # ========================================================
    # 3. LEFT LEG
    # ========================================================

    left_leg_polygon = create_leg_corridor(
        left_hip,
        left_knee,
        left_ankle,

        hip_width=hip_distance * 0.62,
        knee_width=hip_distance * 0.43,
        ankle_width=hip_distance * 0.34
    )

    cv2.fillPoly(
        mask,
        [left_leg_polygon],
        255
    )

    # ========================================================
    # 4. RIGHT LEG
    # ========================================================

    right_leg_polygon = create_leg_corridor(
        right_hip,
        right_knee,
        right_ankle,

        hip_width=hip_distance * 0.62,
        knee_width=hip_distance * 0.43,
        ankle_width=hip_distance * 0.34
    )

    cv2.fillPoly(
        mask,
        [right_leg_polygon],
        255
    )

    # ========================================================
    # 5. CONNECT UPPER THIGHS
    #
    # Adds the area between the two legs but keeps it
    # restricted around the hips/thighs.
    # ========================================================

    inner_thigh_polygon = np.array(
        [
            [
                int(left_hip[0]),
                int(left_hip[1])
            ],
            [
                int(right_hip[0]),
                int(right_hip[1])
            ],
            [
                int(right_knee[0]),
                int(right_knee[1])
            ],
            [
                int(left_knee[0]),
                int(left_knee[1])
            ]
        ],
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [inner_thigh_polygon],
        255
    )

    # ========================================================
    # 6. USE MEDIA PIPE SEGMENTATION AS A SAFETY BOUNDARY
    #
    # Important:
    # We use segmentation ONLY to remove areas outside the
    # person's body.
    #
    # We do NOT use a huge segmentation mask as the final
    # lower-body mask.
    # ========================================================

    if result.segmentation_mask is not None:

        segmentation = (
            result.segmentation_mask > 0.45
        ).astype(np.uint8) * 255

        # Small morphological cleanup
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        segmentation = cv2.morphologyEx(
            segmentation,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Keep only our carefully designed lower-body region
        mask = np.where(
            segmentation > 0,
            mask,
            0
        ).astype(np.uint8)

    # ========================================================
    # 7. REMOVE EVERYTHING ABOVE HIP REGION
    #
    # This prevents the pants mask from accidentally entering
    # the shirt/arms area.
    # ========================================================

    safe_top = int(
        hip_y - hip_distance * 0.22
    )

    safe_top = max(
        0,
        safe_top
    )

    mask[:safe_top, :] = 0

    # ========================================================
    # 8. REMOVE EXTREME OUTER SIDES
    #
    # This protects against bags/accessories near the legs.
    # ========================================================

    body_center_x = int(
        hip_center[0]
    )

    max_side = int(
        hip_distance * 1.15
    )

    outer_left = max(
        0,
        body_center_x - max_side
    )

    outer_right = min(
        width,
        body_center_x + max_side
    )

    side_mask = np.zeros_like(mask)

    side_mask[:, outer_left:outer_right] = 255

    mask = cv2.bitwise_and(
        mask,
        side_mask
    )

    # ========================================================
    # 9. CLEAN MASK
    # ========================================================

    kernel_small = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_small
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    # Slight blur for smoother clothing boundary
    pil_mask = Image.fromarray(
        mask
    )

    pil_mask = pil_mask.filter(
        ImageFilter.GaussianBlur(
            radius=1.2
        )
    )

    # Convert to clean binary mask
    pil_mask = pil_mask.point(
        lambda value:
        255 if value >= 80 else 0
    )

    return pil_mask


# ============================================================
# RUN IDM-VTON
# ============================================================

def run_idm_vton(
    person_image,
    garment_image,
    garment_description,
    custom_mask=None,
    use_crop=True
):

    person_image = Path(
        person_image
    )

    garment_image = Path(
        garment_image
    )

    if not person_image.exists():
        raise FileNotFoundError(
            f"Person image not found:\n{person_image}"
        )

    if not garment_image.exists():
        raise FileNotFoundError(
            f"Garment image not found:\n{garment_image}"
        )

    print("Connecting to IDM-VTON...")

    client = get_vton_client()

    print("Connected successfully.")

    # ========================================================
    # AUTOMATIC MASK
    # ========================================================

    if custom_mask is None:

        human_editor = {
            "background": handle_file(
                str(person_image)
            ),
            "layers": [],
            "composite": None
        }

        use_auto_mask = True

    # ========================================================
    # CUSTOM MASK
    # ========================================================

    else:

        custom_mask = Path(
            custom_mask
        )

        if not custom_mask.exists():
            raise FileNotFoundError(
                f"Mask not found:\n{custom_mask}"
            )

        human_editor = {
            "background": handle_file(
                str(person_image)
            ),
            "layers": [
                handle_file(
                    str(custom_mask)
                )
            ],
            "composite": None
        }

        use_auto_mask = False

    print(
        f"Garment: {garment_image.name}"
    )

    print(
        f"Description: {garment_description}"
    )

    print(
        f"Automatic mask: {use_auto_mask}"
    )

    print(
        f"Crop enabled: {use_crop}"
    )

    print(
        "Generating virtual try-on..."
    )

    result = client.predict(
        human_editor,
        handle_file(
            str(garment_image)
        ),
        garment_description,
        use_auto_mask,
        use_crop,
        30,
        42,
        api_name="/tryon"
    )

    generated_image = result[0]

    masked_image = result[1]

    return (
        generated_image,
        masked_image
    )


# ============================================================
# COMPLETE VIRTUAL TRY-ON
# ============================================================

def generate_virtual_tryon(
    person_image,
    garment_image,
    garment_description="clothing item"
):

    person_image = Path(
        person_image
    )

    garment_image = Path(
        garment_image
    )

    if not person_image.exists():
        raise FileNotFoundError(
            f"Person image not found:\n{person_image}"
        )

    if not garment_image.exists():
        raise FileNotFoundError(
            f"Garment image not found:\n{garment_image}"
        )

    # --------------------------------------------------------
    # W007 -> W007_top.jpg / W007_bottom.jpg
    # --------------------------------------------------------

    outfit_id = garment_image.stem

    top_image = (
        garment_image.parent /
        f"{outfit_id}_top.jpg"
    )

    bottom_image = (
        garment_image.parent /
        f"{outfit_id}_bottom.jpg"
    )

    # PNG fallback
    if not top_image.exists():

        png_top = (
            garment_image.parent /
            f"{outfit_id}_top.png"
        )

        if png_top.exists():
            top_image = png_top

    if not bottom_image.exists():

        png_bottom = (
            garment_image.parent /
            f"{outfit_id}_bottom.png"
        )

        if png_bottom.exists():
            bottom_image = png_bottom

    final_file = (
        OUTPUT_DIR /
        "tryon_result.png"
    )

    final_mask_file = (
        OUTPUT_DIR /
        "tryon_mask.png"
    )

    # ========================================================
    # FULL OUTFIT
    # ========================================================

    if top_image.exists() and bottom_image.exists():

        print("=" * 60)
        print("FULL OUTFIT VIRTUAL TRY-ON")
        print("=" * 60)

        print(
            f"Person: {person_image}"
        )

        print(
            f"Top: {top_image}"
        )

        print(
            f"Bottom: {bottom_image}"
        )

        # ====================================================
        # STAGE 1 — TOP
        # ====================================================

        print(
            "Stage 1/2: Applying top..."
        )

        top_result, top_mask = run_idm_vton(
            person_image=person_image,
            garment_image=top_image,
            garment_description=(
                "olive green long sleeve "
                "fitted cropped top"
            ),
            custom_mask=None,
            use_crop=True
        )

        stage1_file = (
            OUTPUT_DIR /
            "stage1_top.png"
        )

        stage1_mask_file = (
            OUTPUT_DIR /
            "stage1_top_mask.png"
        )

        copy2(
            top_result,
            stage1_file
        )

        copy2(
            top_mask,
            stage1_mask_file
        )

        print(
            f"Stage 1 result saved: {stage1_file}"
        )

        # ====================================================
        # STAGE 2 — LOWER BODY MASK
        # ====================================================

        print(
            "Stage 2/2: Building narrow lower-body mask..."
        )

        lower_mask = build_lower_body_mask(
            stage1_file
        )

        lower_mask_file = (
            OUTPUT_DIR /
            "lower_body_mask.png"
        )

        lower_mask.save(
            lower_mask_file
        )

        print(
            f"Lower-body mask saved: "
            f"{lower_mask_file}"
        )

        # ====================================================
        # STAGE 2 — PANTS
        #
        # IMPORTANT:
        # use_crop=False for the pants stage.
        # This prevents the second generation from moving/
        # distorting the whole person image.
        # ====================================================

        print(
            "Stage 2/2: Applying pants..."
        )

        bottom_result, bottom_mask = run_idm_vton(
            person_image=stage1_file,
            garment_image=bottom_image,
            garment_description=(
                "olive green wide leg pants"
            ),
            custom_mask=lower_mask_file,
            use_crop=False
        )

        copy2(
            bottom_result,
            final_file
        )

        copy2(
            bottom_mask,
            final_mask_file
        )

        print("=" * 60)
        print(
            "FULL OUTFIT VIRTUAL TRY-ON COMPLETED"
        )
        print(
            f"Final result: {final_file}"
        )
        print(
            f"Final mask: {final_mask_file}"
        )
        print("=" * 60)

        return (
            final_file,
            final_mask_file
        )

    # ========================================================
    # SINGLE GARMENT
    # ========================================================

    print("=" * 60)
    print(
        "SINGLE GARMENT VIRTUAL TRY-ON"
    )
    print("=" * 60)

    generated_image, masked_image = run_idm_vton(
        person_image=person_image,
        garment_image=garment_image,
        garment_description=garment_description,
        custom_mask=None,
        use_crop=True
    )

    copy2(
        generated_image,
        final_file
    )

    copy2(
        masked_image,
        final_mask_file
    )

    print(
        f"Result saved: {final_file}"
    )

    print(
        f"Mask saved: {final_mask_file}"
    )

    return (
        final_file,
        final_mask_file
    )