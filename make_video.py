from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
import time

def add_timestamp_to_image(image_path, time_str):
    # 開啟圖片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 設定字體和大小
    try:
        # 嘗試使用系統字體
        font = ImageFont.truetype("Arial", 80)
    except:
        # 如果找不到字體，使用預設字體
        font = ImageFont.load_default()
    
    # 設定文字內容和位置
    text = f"Time Period: {time_str}"
    text_position = (200, 20)  # 左上角位置
    
    # 添加文字陰影效果
    # shadow_offset = 2
    # draw.text((text_position[0] + shadow_offset, text_position[1] + shadow_offset), 
    #           text, fill='black', font=font)
    draw.text(text_position, text, fill='black', font=font)
    
    # 儲存修改後的圖片
    img.save(image_path)

def html_to_images(html_folder="map_outputs", img_folder="map_images"):
    import chromedriver_autoinstaller
    chromedriver_autoinstaller.install()

    html_folder = Path(html_folder)
    img_folder = Path(img_folder)
    img_folder.mkdir(exist_ok=True)

    # 設定 headless 模式的 Chrome
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1280,960")
    driver = webdriver.Chrome(options=options)

    for html_file in tqdm(sorted(html_folder.glob("map_*.html"))):
        url = "file://" + str(html_file.resolve())
        driver.get(url)
        time.sleep(0.5)  # 等待地圖載入
        screenshot_path = img_folder / (html_file.stem + ".png")
        driver.save_screenshot(str(screenshot_path))
        
        # 從檔名中提取時間字串
        time_str = html_file.stem.split("_")[1]
        # 在圖片上添加時間戳記
        add_timestamp_to_image(screenshot_path, time_str)

    driver.quit()
    print("🖼 所有 HTML 地圖已轉成 PNG")

def images_to_gif(img_folder="map_images", output_gif="youbike_animation.gif", duration=1000):
    img_folder = Path(img_folder)
    images = sorted(img_folder.glob("map_*.png"))
    frames = [Image.open(img) for img in images]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # 每張幾毫秒
        loop=0
    )
    print(f"🎞 已輸出 GIF：{output_gif}")

if __name__ == "__main__":
    html_to_images()
    images_to_gif()