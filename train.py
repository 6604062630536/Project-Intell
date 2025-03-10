import pandas as pd
import numpy as np
import random

# โหลดข้อมูล Pokémon (ตัวอย่างนี้ใช้ข้อมูลทั้งหมดหรืออาจกรองเฉพาะ Gen 1-3 หากมีคอลัมน์ Generation)
pokemon_data = pd.read_csv("pokemon.csv")
# ถ้าต้องการเฉพาะ Gen 1-3 uncomment บรรทัดด้านล่าง:
# pokemon_data = pokemon_data[pokemon_data['Generation'] <= 3]

# เลือกเฉพาะฟีเจอร์ที่จำเป็น
pokemon_data = pokemon_data[['Name', 'Total', 'Type 1', 'Type 2', 'Speed']]

# โหลดข้อมูล Type Effectiveness
type_effectiveness_data = pd.read_csv("type_effectiveness.csv")

# สร้าง dictionary สำหรับ Type Effectiveness
type_effectiveness = {}
for _, row in type_effectiveness_data.iterrows():
    attacking_type = row['Attacking Type']
    defending_type = row['Defending Type']
    effectiveness = row['Effectiveness']
    if attacking_type not in type_effectiveness:
        type_effectiveness[attacking_type] = {}
    type_effectiveness[attacking_type][defending_type] = effectiveness

# สร้างคู่การต่อสู้ของทุกคู่ (ไม่จับคู่กับตัวเอง)
battle_data = []
num_pokemon = len(pokemon_data)

for i in range(num_pokemon):
    p1 = pokemon_data.iloc[i]
    for j in range(i+1, num_pokemon):
        p2 = pokemon_data.iloc[j]
        
        # ดึงฟีเจอร์ของแต่ละ Pokémon
        total_p1 = p1["Total"]
        total_p2 = p2["Total"]
        speed_p1 = p1["Speed"]
        speed_p2 = p2["Speed"]
        p1_type1 = p1["Type 1"]
        p1_type2 = p1["Type 2"]
        p2_type1 = p2["Type 1"]
        p2_type2 = p2["Type 2"]
        
        # คำนวณ Type Advantage สำหรับ p1 ต่อ p2
        effectiveness_p1 = 1.0
        if p2_type1 in type_effectiveness.get(p1_type1, {}):
            effectiveness_p1 *= type_effectiveness[p1_type1][p2_type1]
        if p2_type2 in type_effectiveness.get(p1_type1, {}):
            effectiveness_p1 *= type_effectiveness[p1_type1][p2_type2]
        if p2_type1 in type_effectiveness.get(p1_type2, {}):
            effectiveness_p1 *= type_effectiveness[p1_type2][p2_type1]
        if p2_type2 in type_effectiveness.get(p1_type2, {}):
            effectiveness_p1 *= type_effectiveness[p1_type2][p2_type2]
        
        # คำนวณ Speed Advantage
        if speed_p1 > speed_p2:
            speed_advantage = 1.0   # p1 โจมตีก่อน
        elif speed_p1 < speed_p2:
            speed_advantage = 0.0   # p2 โจมตีก่อน
        else:
            speed_advantage = 0.5   # เสมอกัน
        
        # เงื่อนไข: หาก total ของ p1 มากกว่า p2 ถึง 1.3 เท่า (30% มากกว่า) ให้ p1 ชนะทันที
        # หรือหาก total ของ p2 มากกว่า p1 ถึง 1.3 เท่า ให้ p2 ชนะทันที
        if total_p1 >= 1.3 * total_p2:
            win_probability = 1.0
            winner_label = 1  # หมายความว่า p1 ชนะ
        elif total_p2 >= 1.3 * total_p1:
            win_probability = 0.0
            winner_label = 0  # หมายความว่า p2 ชนะ
        else:
            # คำนวณโอกาสชนะจากข้อมูล (ใช้สมการที่คำนวณจาก Stat, Type และ Speed)
            win_probability = (total_p1 * effectiveness_p1 * speed_advantage) / (total_p1 * effectiveness_p1 * speed_advantage + total_p2)
            # กำหนด winner_label จากการสุ่มตาม win_probability
            winner_label = 1 if np.random.rand() < win_probability else 0
        
        battle_data.append([
            total_p1, p1["Type 1"], p1["Type 2"], speed_p1,
            total_p2, p2["Type 1"], p2["Type 2"], speed_p2,
            winner_label, win_probability
        ])

# ลดจำนวนคู่การต่อสู้ลงให้เหลือประมาณ 50,000-60,000 คู่
target_num = 60000
if len(battle_data) > target_num:
    battle_data = random.sample(battle_data, target_num)

# สร้าง DataFrame ด้วยฟีเจอร์ที่ต้องการ
columns = [
    "Total_p1", "Type1_p1", "Type2_p1", "Speed_p1",
    "Total_p2", "Type1_p2", "Type2_p2", "Speed_p2",
    "Winner", "Win_Probability"
]
battle_df = pd.DataFrame(battle_data, columns=columns)

# บันทึกเป็น CSV
battle_df.to_csv("pokemon_battles_features.csv", index=False)

print("✅ สร้างไฟล์ pokemon_battles_features.csv สำเร็จ!")
