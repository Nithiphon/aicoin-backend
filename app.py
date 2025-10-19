from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64

# สร้าง Flask App
app = Flask(__name__)

# แก้ CORS ให้อนุญาตทุก Origin
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# โหลดโมเดล YOLO
print("=" * 50)
print("🔄 กำลังโหลดโมเดล YOLO...")

# ⚠️ เปลี่ยนชื่อไฟล์ตรงนี้ให้ตรงกับไฟล์โมเดลของคุณ
MODEL_PATH = 'model-coin.pt'

try:
    model = YOLO(MODEL_PATH)
    print("✅ โหลดโมเดลสำเร็จ!")
    print(f"📦 ไฟล์โมเดล: {MODEL_PATH}")
    print(f"📋 Classes: {model.names}")
except Exception as e:
    print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    print(f"⚠️  กรุณาตรวจสอบว่าไฟล์ {MODEL_PATH} อยู่ในโฟลเดอร์เดียวกับ app.py")
    exit()

print("=" * 50)

# Route หน้าแรก
@app.route('/')
def home():
    return """
    <h1>🪙 Coin Detection API</h1>
    <p>✅ Server กำลังทำงาน</p>
    <p>📍 ส่งรูปมาที่: POST /detect</p>
    <p>🎯 รองรับเหรียญ: 1, 5, 10 บาท</p>
    """

# Route สำหรับ OPTIONS (CORS preflight)
@app.route('/detect', methods=['OPTIONS'])
def detect_options():
    return '', 204

# Route สำหรับตรวจจับเหรียญ
@app.route('/detect', methods=['POST'])
def detect_coins():
    try:
        print("\n" + "=" * 50)
        print("📥 รับ Request ใหม่")
        
        # ตรวจสอบว่ามีไฟล์รูปส่งมาหรือไม่
        if 'image' not in request.files:
            print("❌ ไม่พบไฟล์รูปภาพ")
            return jsonify({
                'success': False,
                'error': 'ไม่พบไฟล์รูปภาพ'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            print("❌ ไม่ได้เลือกไฟล์")
            return jsonify({
                'success': False,
                'error': 'ไม่ได้เลือกไฟล์'
            }), 400
        
        print(f"📄 ชื่อไฟล์: {file.filename}")
        
        # อ่านรูปภาพ
        print("🔄 กำลังอ่านรูปภาพ...")
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print("❌ ไม่สามารถอ่านรูปภาพได้")
            return jsonify({
                'success': False,
                'error': 'ไม่สามารถอ่านรูปภาพได้ กรุณาตรวจสอบไฟล์'
            }), 400
        
        print(f"📐 ขนาดรูป: {img.shape}")
        
        # ประมวลผลด้วย YOLO
        print("🤖 กำลังประมวลผลด้วย YOLO...")
        results = model(img, conf=0.5, verbose=False)
        
        # ดึงข้อมูลการตรวจจับ
        detections = results[0].boxes
        coin_count = len(detections)
        
        # นับจำนวนเหรียญแต่ละชนิด
        coin_types = {
            '1baht': 0,
            '5baht': 0,
            '10baht': 0
        }
        
        total_value = 0
        
        print(f"✅ พบเหรียญทั้งหมด: {coin_count} เหรียญ")
        
        if coin_count > 0:
            print("📊 รายละเอียด:")
            
            for i, box in enumerate(detections):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                
                print(f"   เหรียญที่ {i+1}: {label} (ความมั่นใจ {conf:.2%})")
                
                # นับจำนวนและคำนวณมูลค่า
                if label == "1baht":
                    coin_types['1baht'] += 1
                    total_value += 1
                elif label == "5baht":
                    coin_types['5baht'] += 1
                    total_value += 5
                elif label == "10baht":
                    coin_types['10baht'] += 1
                    total_value += 10
        
        print(f"💰 มูลค่ารวม: {total_value} บาท")
        
        # วาดกรอบบนรูปภาพ
        img_with_boxes = results[0].plot()
        
        # แปลงรูปเป็น base64 เพื่อส่งกลับไป
        _, buffer = cv2.imencode('.jpg', img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print("=" * 50 + "\n")
        
        # ส่งผลลัพธ์กลับ
        return jsonify({
            'success': True,
            'total_count': coin_count,
            'total_value': total_value,
            'coin_details': {
                '1baht': coin_types['1baht'],
                '5baht': coin_types['5baht'],
                '10baht': coin_types['10baht']
            },
            'image_with_boxes': img_base64,
            'message': f'พบเหรียญทั้งหมด {coin_count} เหรียญ มูลค่ารวม {total_value} บาท'
        })
    
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 50 + "\n")
        return jsonify({
            'success': False,
            'error': f'เกิดข้อผิดพลาด: {str(e)}'
        }), 500

# เริ่มต้น Flask Server
if __name__ == '__main__':
    # รับ PORT จาก Environment Variable (สำหรับ Deploy)
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "🚀" * 25)
    print("🚀 เริ่มต้น Coin Detection Server")
    print(f"📍 PORT: {port}")
    print("🪙 รองรับเหรียญ: 1, 5, 10 บาท")
    print("🌐 กด Ctrl+C เพื่อหยุด Server")
    print("🚀" * 25 + "\n")
    
    # debug=False สำหรับ Production
    app.run(debug=False, port=port, host='0.0.0.0')