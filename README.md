# โปรแกรมตรวจจับความเสียหายของถนนด้วย FRCNN
# Ver 1.0
- อ่าน GPS 
- หาความเร็วใน 1 วินาที
- ตัดภาพ 1 วินาที/ 1 แฟรม
- ถ้าขี่เร็วเกิน 50 km/h โปรแกรมจะตัดภาพ 1 วินาที/ 2 แฟรม
- ถ้าใช้ 1 วินาที/ 2 แฟรม จะทำการ linear interpolate GPS ที่ 0.5 วินาที
- หาระยะทางทั้งหมดของ GPS
- เพิ่มการ Plot ความเสียหาย
- ทราบความเสียหายอยู่ซ้าย หรือ ขวาของวิดีโอ 
- เมื่อหาหลุมเจอจะตัดภาพหลุมออกมาเก็บไว้ทั้งหมด

![img](https://i.imgur.com/quLHqqw.png)

# ทดสอบด้วยเกม GTA V
- ถนน
![img](https://i.imgur.com/aM39tIz.jpg)

# นำวิดีโอเข้า
- ตรวจสอบด้วยวิดีโอ 1 วินาทีต่อ 6 ภาพ
![img](https://i.imgur.com/n3drsiU.png)

# เปรียบเทียบกับ GTA V Map
- จุดจะมีมากหรือน้อย ขึ้นอยู่กับความเร็วเฟรมที่เลือก
![img](https://i.imgur.com/bAZJXzk.jpg)

# นำจุดทั้งหมดสร้างเป็นกราฟความร้อน (ความเสียหาย)
![img](https://i.imgur.com/lnLOdee.png)

# นำข้อมูลทั้งหมดมารวมกัน
![img](https://media0.giphy.com/media/p9jDAX9JruRnaYmcLx/giphy.gif?cid=790b761187dc30cbf0d00a37305b351c75943164c9a530ce&rid=giphy.gif&ct=g)



