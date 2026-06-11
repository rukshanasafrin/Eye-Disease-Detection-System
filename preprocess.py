import cv2

def preprocess_image(image_path, save_path=None):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Save enhanced image (optional)
    if save_path:
        cv2.imwrite(save_path, enhanced_img)

    # Convert to RGB for model compatibility
    rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    return rgb_img
