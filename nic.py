import time
import psutil


# 获取网络接口的计数器信息
def get_network_counters(interface):
    counters = psutil.net_io_counters(pernic=True)
    if interface in counters:
        return counters[interface]
    else:
        raise ValueError("Invalid network interface")


# 监控网络接口计数器
def monitor_network_counters(interface, duration):
    start_counters = get_network_counters(interface)
    start_time = time.time()

    while True:
        time.sleep(1)
        current_counters = get_network_counters(interface)
        elapsed_time = time.time() - start_time

        # 计算流量速率
        rx_bytes = current_counters.bytes_recv - start_counters.bytes_recv
        tx_bytes = current_counters.bytes_sent - start_counters.bytes_sent
        rx_speed = rx_bytes / elapsed_time
        tx_speed = tx_bytes / elapsed_time

        # 输出结果
        print(f"时间: {elapsed_time:.2f}秒")
        print(f"接收: {rx_speed:.2f} bytes/秒")
        print(f"发送: {tx_speed:.2f} bytes/秒")

        # 检查是否达到指定的监控时长
        if elapsed_time >= duration:
            break


# 主程序
if __name__ == "__main__":
    interface = "eth0"  # 更改为要监控的网络接口名称
    duration = 60  # 监控时长（秒）

    try:
        monitor_network_counters(interface, duration)
    except ValueError as e:
        print(f"错误: {e}")
