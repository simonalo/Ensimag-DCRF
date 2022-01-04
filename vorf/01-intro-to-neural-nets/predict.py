from model import config
import sys
import os
import torch
import cv2

# load our object detector, set it evaluation mode, and label
# encoder from disk
print("**** loading object detector...")
model = torch.load(config.BEST_MODEL_PATH).to(config.DEVICE)
model.eval()

data = []
for path in sys.argv[1:]:
    if path.endswith('.csv'):
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(path).read().strip().split("\n"):
            filename, startX, startY, endX, endY, label = row.split(',')
            filename = os.path.join(config.IMAGES_PATH, label, filename)
            data.append((filename, startX, startY, endX, endY, label))
    elif path == "-f":
        pass
    else:
        data.append((path, None, None, None, None, None))

# -f to loop through bad predictions
bad_pred = sys.argv[1] == "-f"

# loop over images to be tested with our model, with ground truth if available
for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
    # load the image, copy it, swap its colors channels, resize it, and
    # bring its channel dimension forward
    image = cv2.imread(filename)
    display = image.copy()
    h, w = display.shape[:2]

    # convert image to PyTorch tensor, normalize it, upload it to the
    # current device, and add a batch dimension
    image = config.TRANSFORMS(image).to(config.DEVICE)
    image = image.unsqueeze(0)

    # predict the bounding box of the object along with the class label
    label_predictions, bbox_predictions = model(image)

    # determine the class label with the largest predicted probability
    label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
    most_likely_label = label_predictions.argmax(dim=-1).cpu()
    label = config.LABELS[most_likely_label]

    # Denormalize bounding box from (0,1)x(0,1) to (0,w)x(0,h)
    bbox_startX = bbox_predictions[0][0] * w
    bbox_startY = bbox_predictions[0][1] * h
    bbox_endX = bbox_predictions[0][2] * w
    bbox_endY = bbox_predictions[0][3] * h

    # draw the ground truth box and class label on the image, if any
    if gt_label is not None:
        cv2.putText(display, 'gt ' + gt_label, (0, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0,  0), 2)
        # Display ground truth bounding box in blue
        cv2.rectangle(display, (int(gt_start_x), int(gt_start_y)), (int(gt_end_x), int(gt_end_y)), (255, 0, 0), 2)

    # draw the predicted bounding box and class label on the image
    cv2.putText(display, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    # Display predicted bounding box
    cv2.rectangle(display, (int(bbox_startX), int(bbox_startY)), (int(bbox_endX), int(bbox_endY)), (0, 0, 255), 2)

    if label != gt_label or not bad_pred:
        # show the output image
        print(bbox_predictions)
        cv2.imshow("Output", display)
        key = cv2.waitKey(0)
        if key == 27:
            break
