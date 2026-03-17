import cv2
import os


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def binarize_image(input_path, output_path):
    """
    对图像进行二值化处理，使背景变白，文字变黑。
    使用自适应阈值处理，能够较好地处理光照不均的情况。
    """
    # 1. 以灰度模式读取图片
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法读取图片 {input_path}")
        return

    # 2. 高斯滤波去噪
    # 较大的 kernel 可以平滑背景，但可能会模糊文字，这里取 (5, 5)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. 自适应阈值二值化 (Adaptive Thresholding)
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 阈值是邻域区域的加权和
    # cv2.THRESH_BINARY: 超过阈值的设为 255 (白色), 否则设为 0 (黑色)
    # block_size 和 C 的取值根据实际情况微调
    # block_size 必须是奇数，如 11, 21, 31...
    # C 是从均值或加权均值中减去的常数，用于控制结果的亮暗程度
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        21, 
        10
    )

    # 4. 保存结果
    cv2.imwrite(output_path, binary)
    print(f"二值化完成，结果已保存至: {output_path}")

def main():
    input_dir = r"D:\tools\313-toolbox\img-binarization\input"
    output_dir = r"D:\tools\313-toolbox\img-binarization\output"

    if not os.path.isdir(input_dir):
        print(f"错误: 输入文件夹 {input_dir} 不存在，请创建后再运行。")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"提示: 输入文件夹 {input_dir} 中没有可处理的图片。")
        return

    for image_name in image_files:
        input_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        binarize_image(input_path, output_path)

    print(f"批量处理完成，共处理 {len(image_files)} 张图片。")

if __name__ == "__main__":
    main()
