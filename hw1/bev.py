import cv2
import numpy as np
import math

points = []
new_pixels = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=1.5, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        
        #innitial the matrix
        fy = math.tan(fov/2 * np.pi/180) * (self.height/2) 
        fx = math.tan((fov/2) * np.pi/180) * (self.width/2)
        cos_y = math.cos(theta * np.pi/180)
        sin_y = np.sin(theta * np.pi/180)
        cos_b = np.cos(phi * np.pi/180)
        sin_b = np.sin(phi * np.pi/180)
        cos_a = np.cos(gamma * np.pi/180) 
        sin_a = np.sin(gamma * np.pi/180)
        transform_matrix = np.array([[(cos_a*cos_b), (cos_a*sin_b*sin_y)-(sin_a*cos_y), (cos_a*sin_b*cos_y)+(sin_a*sin_y), dx],
                                     [(sin_a*cos_b), (sin_a*sin_b*sin_y)+(cos_a*cos_y), (sin_a*sin_b*cos_y)-(cos_a*sin_y), dy],
                                     [(-sin_b)     , (cos_b*sin_y)                    , (cos_b*cos_y)                    , dz],
                                     [0,0,0,1]]
                                    )
        print(transform_matrix)
        for point in points:
            u = point[0]
            v = point[1]

            #bev pixel to camera center
            pixel_x = -u + (self.width / 2)
            pixel_y = -v + (self.height / 2)
            print("-----------------------------------")
            print("bev pixel to camera center\n")
            print(pixel_x, pixel_y)
                    
            #camera center to bev point
            bev_x = (pixel_x / fx) * 2.5
            bev_y = (pixel_y / fy) * 2.5 
            bev_z = 2.5
            bev_point = np.array([bev_x, bev_y, bev_z, 1])
            print("-----------------------------------")
            print("camera center to bev point\n")
            print(bev_point)
            
            #bev to front
            front_point = np.dot(transform_matrix, bev_point.T)
            front_point = front_point.T
            print("-----------------------------------")
            print("bev to front\n")
            print(front_point)

            #front point to camera center
            front_z = front_point[2]
            pixel_x = (front_point[0]/front_z) * fx
            pixel_y = (front_point[1]/front_z) * fy
            print("-----------------------------------")
            print("front point to camera center\n")
            print(pixel_x, pixel_y)
            
            #camera center to front pixel
            new_pixels_x = -pixel_x + (self.width/2)
            new_pixels_y = -pixel_y + (self.width/2)
            new_pixels.append([int(new_pixels_x),int(new_pixels_y)])
            print("-----------------------------------")
            print("camera center to front pixel\n")
            print(new_pixels)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

if __name__ == "__main__":

    pitch_ang = 90

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):  #press q to quit
            break
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
