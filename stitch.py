import cv2
import numpy as np
import os
import sys

class Matchers:
    """
    A class to handle feature matching between images using SIFT and FLANN
    """
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create() 
        # FLANN parameters for SIFT matching
        FLANN_INDEX_KDTREE = 1  # Using FLANN_INDEX_KDTREE for SIFT
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, img1, img2, direction=None):
        """
        Match features between two images
        Args:
            img1, img2: Input images to be matched
            direction: Stitching direction (left/right)
        Returns:
            H: Homography matrix for image transformation
        """
        # Extract SIFT features from both images
        imageSet1 = self.getSIFTFeatures(img1)  
        imageSet2 = self.getSIFTFeatures(img2) 
        print("Direction:", direction)
        
        # Find k-nearest matches for each descriptor
        matches = self.flann.knnMatch(imageSet2['des'], imageSet1['des'], k=2)
        good = []
        # Apply ratio test to filter good matches
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        # Calculate homography if enough good matches are found
        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (_, i) in good])
            matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, _) in good])

            # Find homography using RANSAC
            H, _ = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
            return H
        return None

    def getSIFTFeatures(self, im):
        """
        Extract SIFT features from an image
        Args:
            im: Input image
        Returns:
            Dictionary containing keypoints and descriptors
        """
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

class Stitch:
    """
    Main class for image stitching operations
    """
    def __init__(self, image_folder):
        """
        Initialize the stitching process
        Args:
            image_folder: Path to folder containing images to be stitched
        """
        self.images = self.load_images(image_folder)
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = Matchers()
        self.prepare_lists()

    def load_images(self, folder):
        """
        Load and resize all images from the specified folder
        Args:
            folder: Path to image folder
        Returns:
            List of resized images
        """
        filenames = [f for f in os.listdir(folder) if f.endswith(('jpg', 'jpeg', 'png', 'JPG'))]
        images = [cv2.resize(cv2.imread(os.path.join(folder, f)), (480, 320)) for f in filenames]
        return images

    def prepare_lists(self):
        """
        Prepare image lists for left and right stitching
        Divides images into left and right lists based on center image
        """
        print(f"Number of images: {self.count}")
        self.centerIdx = self.count // 2
        print(f"Center index image: {self.centerIdx}")
        self.center_im = self.images[self.centerIdx]
        self.left_list = self.images[:self.centerIdx + 1]
        self.right_list = self.images[self.centerIdx + 1:]
        print("Image lists prepared")

    def leftshift(self):
        """
        Stitch images to the left of the center image
        Applies perspective transformation and combines images
        """
        # Start with the leftmost image in the left list
        a = self.left_list[0]
        
        # Iterate through remaining images from left to center
        for b in self.left_list[1:]:
            # Find homography matrix between consecutive images
            H = self.matcher_obj.match(a, b, 'left')
            print("Homography is:", H)
            
            # Calculate inverse homography matrix for perspective transform
            xh = np.linalg.inv(H)
            
            # Calculate dimensions of warped image using corner points
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds / ds[-1]  # Normalize homogeneous coordinates
            
            # Calculate offset for the origin point (0,0)
            f1 = np.dot(xh, np.array([0, 0, 1]))
            
            # Adjust homography matrix to handle negative offsets
            xh[0][-1] += abs(f1[0])  # Add x-offset to transformation
            xh[1][-1] += abs(f1[1])  # Add y-offset to transformation
            
            # Recalculate final image dimensions after offset adjustment
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            
            # Calculate absolute offsets for image placement
            offsety = abs(int(f1[1]))  # Vertical offset
            offsetx = abs(int(f1[0]))  # Horizontal offset
            
            # Define size of output image including offsets
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
            print("Image dsize =>", dsize)
            
            # Apply perspective transform to first image
            tmp = cv2.warpPerspective(a, xh, dsize)
            
            # Copy second image onto the transformed first image
            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
            
            # Update working image for next iteration
            a = tmp
        
        # Store final left-stitched image
        self.leftImage = tmp
    
    def rightshift(self):
        """
        Stitch images to the right of the center image
        Applies perspective transformation and combines images with blending
        """
        # Process each image in right list
        for each in self.right_list:
            # Find homography matrix between left image and current right image
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            print("Homography:", H)
            
            # Get dimensions of current left image
            h, w = self.leftImage.shape[:2]
            
            # Calculate new image dimensions after transformation
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]  # Normalize homogeneous coordinates
            
            # Calculate size of new canvas needed
            dsize = (int(txyz[0]) + self.leftImage.shape[1], int(txyz[1]) + self.leftImage.shape[0])
            
            # Apply perspective transform to right image
            tmp = cv2.warpPerspective(each, H, dsize)
            
            # Create new blank canvas for composite image
            new_img = np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)
            
            # Copy left image to new canvas
            new_img[:h, :w] = self.leftImage
            
            # Blend the overlapping regions
            result = self.mix_and_match(new_img, tmp)
            
            # Find valid region in stitched image by creating binary mask
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Find contours of valid image region
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Crop to valid region if contours found
            if contours:
                # Find bounding rectangle of largest contour
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                self.leftImage = result[y:y+h, x:x+w]
            else:
                self.leftImage = result

    def mix_and_match(self, leftImage, warpedImage):
        """
        Blend two images in their overlapping region
        Args:
            leftImage: Left side image
            warpedImage: Transformed right side image
        Returns:
            result: Blended image
        """
        # Create masks for overlap region
        gray_left = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
        
        # Create binary masks
        _, mask_left = cv2.threshold(gray_left, 1, 255, cv2.THRESH_BINARY)
        _, mask_right = cv2.threshold(gray_right, 1, 255, cv2.THRESH_BINARY)
        
        # Find overlapping region
        overlap = cv2.bitwise_and(mask_left, mask_right)
        
        # Dilate overlap region
        kernel_size = 100  # Increase kernel size for smoother transition
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        overlap_dilated = cv2.dilate(overlap, kernel)
        
        # Create weight map
        rows, cols = overlap_dilated.shape
        distances = np.zeros((rows, cols))
        
        # Calculate distances to edges
        for y in range(rows):
            for x in range(cols):
                if overlap_dilated[y,x] > 0:
                    left_edge = x
                    right_edge = x
                    for i in range(x, -1, -1):
                        if overlap_dilated[y,i] == 0:
                            left_edge = i
                            break
                    for i in range(x, cols):
                        if overlap_dilated[y,i] == 0:
                            right_edge = i
                            break
                    # Calculate relative distance as weight
                    if right_edge > left_edge:
                        distances[y,x] = float(x - left_edge) / float(right_edge - left_edge)
        
        # Smooth weight map
        distances = cv2.GaussianBlur(distances, (51,51), 0)
        
        # Create three-channel weights
        weight_right = cv2.merge([distances, distances, distances])
        weight_left = 1 - weight_right
        
        # Apply weights
        result = leftImage.astype(float) * weight_left + warpedImage.astype(float) * weight_right
        
        # Copy non-overlapping regions directly
        result[mask_left == 0] = warpedImage[mask_left == 0]
        result[mask_right == 0] = leftImage[mask_right == 0]
        
        return result.astype(np.uint8)

    def showImage(self):
        """
        Display the final stitched image
        """
        cv2.imshow("Stitched Image", self.leftImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Specify the folder containing images to be stitched
    image_folder = 'street'
    stitcher = Stitch(image_folder)
    stitcher.leftshift()
    stitcher.rightshift()
    stitcher.showImage()
    cv2.imwrite("stitched_street.jpg", stitcher.leftImage)
    print("Stitched image saved as stitched_output.jpg")