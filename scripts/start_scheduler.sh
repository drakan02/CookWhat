#!/bin/bash
set -e

echo "=== KHỞI ĐỘNG CRAWL SCHEDULER ==="

# 1. Xuất tất cả biến môi trường Docker sang /etc/environment
# Điều này cực kỳ quan trọng vì tiến trình Cron chạy cô lập và không tự thừa kế env variables của container.
env | grep -v 'no_proxy' >> /etc/environment

# 2. Tạo file cấu hình cron job
# 2h sáng thứ 2 hàng tuần: 0 2 * * 1 (tương ứng thứ 2 là 1)
# Trong môi trường Docker container, python nằm tại /usr/local/bin/python
CRON_SCHEDULE="0 2 * * 1"

echo "${CRON_SCHEDULE} cd /app && /usr/local/bin/python scripts/run_periodic_pipeline.py >> /app/cookpad_data/cron.log 2>&1" > /etc/cron.d/cookwhat-cron

# Cấp quyền cho file cấu hình cron và nạp vào crontab daemon
chmod 0644 /etc/cron.d/cookwhat-cron
crontab /etc/cron.d/cookwhat-cron

echo "[Scheduler] Đã cấu hình Cron Job thành công với lịch biểu: '${CRON_SCHEDULE}'"
echo "[Scheduler] Đang khởi chạy cron daemon ở foreground..."

# 3. Khởi chạy cron daemon ở chế độ foreground (-f) để giữ container luôn chạy
exec cron -f
