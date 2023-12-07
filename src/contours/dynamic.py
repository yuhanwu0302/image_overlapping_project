import matplotlib.pyplot as plt

def run():
    
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 800)
    ax.invert_yaxis()
    # 定义按键响应函数
    def on_press(event):
        if event.key == 'escape':  # 如果按下的是 Esc 键
            plt.close(fig)  # 关闭图表窗口

    # 将按键事件与处理函数绑定
    fig.canvas.mpl_connect('key_press_event', on_press)

    # 逐点绘制轮廓
    for i in range(len(clockwise)):
        x, y = clockwise[i, 0], clockwise[i, 1]
        ax.scatter(x, y)
        plt.pause(0.000001)
        if not plt.fignum_exists(fig.number):  # 如果图表已关闭，则结束循环
            break

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.show()

run()