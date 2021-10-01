import cv2 as cv

def get_blocks3D(arr, num_row_blocks=4, num_col_blocks=5):
    """
    :param arr: Array to partition into blocks
    :return: Generates num_row_blocks * num_col_blocks equally-sized pieces of the array
    """

    height = arr.shape[0]
    width = arr.shape[1]

    stride_row = int(width / num_row_blocks)
    stride_col = int(height / num_col_blocks)

    for x in range(0, arr.shape[0], stride_col):
        for y in range(0, arr.shape[1], stride_row):
            #print(f'x = {x}:{x+stride_col}; y = {y}:{y+stride_row}')
            yield arr[x:x + stride_col, y:y + stride_row, :]

def get_blocks2D(arr, num_row_blocks=4, num_col_blocks=5):
    """
    :param arr: Array to partition into blocks
    :return: Generates num_row_blocks * num_col_blocks equally-sized pieces of the array
    """

    height = arr.shape[0]
    width = arr.shape[1]

    stride_row = int(width / num_row_blocks)
    stride_col = int(height / num_col_blocks)

    for x in range(0, arr.shape[0], stride_col):
        for y in range(0, arr.shape[1], stride_row):
            yield arr[x:x + stride_col, y:y + stride_row]


def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=4, pxstep=90, pystep=128):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    x = pxstep
    y = pystep
    while x < img.shape[1]:
        cv.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pystep
    return img