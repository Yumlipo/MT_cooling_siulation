import cv2
from screeninfo import get_monitors
from get_img import mouse_action

# import ctypes 
# def Mbox(title, text, style):
#     return ctypes.windll.user32.MessageBoxW(0, text, title, style)

# message = "    smth\n "
# # Mbox('Program usage rules', message, 1)

SELECT = 0
screen = get_monitors()
# print(screen[0].height)

for monitor in get_monitors():
    work_area = [monitor.width, monitor.height - 100]
    print(str(work_area[0]) + 'x' + str(work_area[1]))

cv2.namedWindow('MT cooling', cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('MT cooling', int(0.5 * work_area[0]), int(0.5 * work_area[1]))


while cv2.getWindowProperty("MT cooling", cv2.WND_PROP_VISIBLE) > 0:
    import get_img
    cv2.setMouseCallback('MT cooling', mouse_action)

    key = cv2.waitKey(0) & 0xFF
    # print(key)

#     # if key == 13:  # If 'enter' is pressed calculate smth


    if key == ord("q"):
        cv2.destroyAllWindows()
        break