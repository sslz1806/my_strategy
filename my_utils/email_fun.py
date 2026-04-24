# 参考上面的邮件发送图片添加到正文
# 现在把report\2024-06-07\gld\下的图片按顺序barr,industry totals 放进html格式的邮件正文,
# strategy_name_2024-06-07.xlsx 放进附件
# 邮件使用腾讯企业邮箱
# 腾讯企业邮箱的IMAP服务器地址
from datetime import datetime, timedelta
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import imaplib
import os
import smtplib
from email.mime.image import MIMEImage
import re
import pandas as pd
import chardet
smtp_host = 'smtp.qq.com'
smtp_port = 465
mail_username = '2056123357@qq.com'
auth_code = 'vikdtadiiypsidaa'  # 请替换为你的邮箱授权码（注意：不是登录密码！）
# user_list = ['luoyajun@lsshs.com']
user_list = ['2056123357@qq.com','1712167056@qq.com','liyi@lsshs.com']
from typing import List, Optional
from email.mime.base import MIMEBase
from email.header import Header
from email import encoders

# 修改为正确的参数顺序
def sendStringEmail(user_list, subject, info="这是一封测试邮件", date=None):
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    msg = MIMEMultipart('related')
    msg["Subject"] = f"{subject} {date}"
    msg["From"] = mail_username
    msg["To"] = mail_username
    # html格式的邮件正文
    content = f'''
    <body>
    <p> {info} </p>
    </body>
    '''
    msg.attach(MIMEText(content, 'html', 'utf-8'))
    print(content)
    s = smtplib.SMTP(smtp_host, smtp_port)
    s.login(mail_username, auth_code)

    s.sendmail(mail_username, user_list, msg.as_string())
    s.quit()
    print("邮件发送成功！")


def send_email(
    subject: str,
    body: str,
    sender_email: str=mail_username,
    sender_auth_code: str=auth_code,
    receiver_emails: List[str]=user_list,
    body_type: str = "plain",  # 可选: "plain" (纯文本) 或 "html" (富文本)
    attachment_paths: Optional[List[str]] = None,
    smtp_server: str = "smtp.qq.com",  # 默认QQ邮箱，其他邮箱见下方说明
    smtp_port: int = 465,  # SSL端口，通常是465或587
    use_ssl: bool = True
) -> bool:
    """
    发送邮件的通用函数，支持文本/HTML正文、附件。

    :param sender_email: 发件人邮箱地址
    :param sender_auth_code: 发件人邮箱授权码（注意：不是登录密码！）
    :param receiver_emails: 收件人邮箱列表（支持多人）
    :param subject: 邮件主题
    :param body: 邮件正文内容
    :param body_type: 正文类型，"plain" (纯文本) 或 "html" (富文本)
    :param attachment_paths: 附件文件路径列表（可选）
    :param smtp_server: SMTP服务器地址（默认QQ邮箱，其他见下方）
    :param smtp_port: SMTP端口（默认465，SSL端口）
    :param use_ssl: 是否使用SSL加密（默认True）
    :return: 发送成功返回True，失败返回False
    """
    # 1. 初始化邮件对象
    if attachment_paths:
        # 有附件时用 MIMEMultipart
        msg = MIMEMultipart()
    else:
        # 无附件时直接用 MIMEText
        msg = MIMEText(body, body_type, "utf-8")

    # 2. 设置邮件头
    msg["From"] = Header(sender_email)
    msg["To"] = Header(",".join(receiver_emails))
    msg["Subject"] = Header(subject, "utf-8")

    # 3. 添加正文（仅当有附件时需要单独添加，无附件时已在初始化时处理）
    if attachment_paths:
        msg.attach(MIMEText(body, body_type, "utf-8"))

        # 4. 添加附件
        for file_path in attachment_paths:
            if not os.path.exists(file_path):
                print(f"警告：附件文件 {file_path} 不存在，跳过。")
                continue

            # 读取附件文件
            with open(file_path, "rb") as f:
                mime = MIMEBase("application", "octet-stream")
                mime.set_payload(f.read())

            # 编码附件
            encoders.encode_base64(mime)

            # 设置附件文件名（处理中文文件名乱码）
            file_name = os.path.basename(file_path)
            mime.add_header(
                "Content-Disposition",
                "attachment",
                filename=("utf-8", "", file_name)
            )

            msg.attach(mime)

    # 5. 连接SMTP服务器并发送
    try:
        if use_ssl:
            # 使用SSL加密连接（推荐，更安全）
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            # 普通连接（不推荐，部分邮箱不支持）
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # 开启TLS加密（如果端口是587）

        # 登录邮箱
        server.login(sender_email, sender_auth_code)

        # 发送邮件
        server.sendmail(sender_email, receiver_emails, msg.as_string())

        # 退出连接
        server.quit()
        print("邮件发送成功！")
        return True

    except smtplib.SMTPAuthenticationError:
        print("错误：邮箱认证失败！请检查发件人邮箱和授权码是否正确。")
        return False
    except smtplib.SMTPConnectError:
        print("错误：无法连接SMTP服务器！请检查SMTP服务器地址和端口是否正确。")
        return False
    except Exception as e:
        print(f"邮件发送失败：{str(e)}")
        return False

if __name__ == '__main__':
    msg = "这是一封测试邮件"
    #sendStringEmail(['2056123357@qq.com'], '测试邮件', msg)
    send_email(
        sender_email=mail_username,
        sender_auth_code=auth_code,
        receiver_emails=user_list,
        subject="测试邮件",
        body=msg,
        body_type="html",
        attachment_paths=None,  # 可添加附件路径列表，如 ["report_20240607.xlsx"]
    )