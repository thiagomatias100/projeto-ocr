import cv2
from pathlib import Path

def preprocess_gray_debug(img_bgr):
    stages = {}

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    stages["01_gray"] = gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    stages["02_clahe"] = clahe

    blur  = cv2.GaussianBlur(clahe, (0,0), 1.0)
    stages["03_blur"] = blur

    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    stages["04_sharp"] = sharp

    return sharp, stages

if __name__ == "__main__":
    path = input("Imagem (.png/.jpg): ").strip().strip('"').strip("'")
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit("não consegui ler a imagem.")
    if img.ndim == 3 and img.shape[2] == 4:  # BGRA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    out_dir = Path("debug_stages"); out_dir.mkdir(exist_ok=True)
    base = Path(path).stem

    final, stages = preprocess_gray_debug(img)
    for name, im in stages.items():
        cv2.imwrite(str(out_dir / f"{base}_{name}.png"), im)

    cv2.imwrite(str(out_dir / f"{base}_05_final_sharp.png"), final)
    print("salvo em:", out_dir.resolve())
