import cv2
import os
import numpy as np

# 显示图像
def cvshow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 全景拼接
def siftimg_rightlignment(img_right, img_left):
    """图像拼接"""
    # 使用新版SIFT
    sift = cv2.SIFT_create()
    
    # 转换为灰度图并提取特征点
    gray1 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # 特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) > 4:
        # 获取匹配点坐标
        ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # 计算变换矩阵
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
        
        # 获取图像尺寸
        h1, w1 = img_right.shape[:2]
        h2, w2 = img_left.shape[:2]
        
        # 计算变换后的边界
        corners = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # 计算边界和偏移
        [xmin, ymin] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
        
        # 确保偏移量为正
        t = [-xmin if xmin < 0 else 0, -ymin if ymin < 0 else 0]
        
        # 计算输出图像大小
        width = xmax - xmin
        height = max(ymax - ymin, h2)
        
        # 更新变换矩阵
        Ht = np.array([[1,0,t[0]], [0,1,t[1]], [0,0,1]])
        H = Ht.dot(H)
        
        # 创建输出图像
        result = cv2.warpPerspective(img_right, H, (width, height))
        
        # 确保坐标有效
        y_offset = t[1]
        x_offset = t[0]
        result[y_offset:y_offset+h2, x_offset:x_offset+w2] = img_left
        
        # 裁剪黑边
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            result = result[y:y+h, x:x+w]
        
        return result
    return None
    
def read_images(folder):
    """读取文件夹中的所有图片"""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def preprocess_images(images):
    """调整所有图片为相同大小"""
    if not images:
        return []
    # 以第一张图片的大小为基准
    height, width = images[0].shape[:2]
    processed = []
    for img in images:
        resized = cv2.resize(img, (width, height))
        processed.append(resized)
    return processed

def stitch_images(images):
    """拼接多张图片"""
    if not images:
        return None
    
    result = images[0]
    for i in range(1, len(images)):
        result = siftimg_rightlignment(images[i], result)
        if result is None:
            print(f"拼接第 {i} 张图片失败")
            return None
    return result
      

def main():
    # 创建输出目录
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 读取图片
    images = read_images('images')
    if not images:
        print("未找到任何图片")
        return
    
    # 预处理图片
    processed_images = preprocess_images(images)
    
    # 拼接图片
    result = stitch_images(processed_images)
    
    if result is not None:
        # 保存结果
        output_path = os.path.join('output', 'panorama.jpg')
        cv2.imwrite(output_path, result)
        print(f"全景图已保存至: {output_path}")
        
        # 显示结果
        cvshow('Final Panorama', result)
    else:
        print("图片拼接失败")

if __name__ == "__main__":
    main()
