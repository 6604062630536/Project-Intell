import pandas as pd
import numpy as np

# โหลดข้อมูล Pokémon
pokemon_data = pd.read_csv("pokemon.csv")

# โหลดข้อมูล Type Effectiveness
type_effectiveness_data = pd.read_csv("type_effectiveness.csv")

# สร้าง dictionary สำหรับ Type Effectiveness
type_effectiveness = {}
for _, row in type_effectiveness_data.iterrows():
    attacking_type = row['Attacking Type']
    defending_type = row['Defending Type']
    effectiveness = row['Effectiveness']
    
    # บันทึกข้อมูลลงใน dictionary
    if attacking_type not in type_effectiveness:
        type_effectiveness[attacking_type] = {}
    
    type_effectiveness[attacking_type][defending_type] = effectiveness

# สุ่มการต่อสู้
num_battles = 10000  # จำนวนการต่อสู้ที่จำลอง
battle_data = []

for _ in range(num_battles):
    p1, p2 = np.random.choice(pokemon_data["Name"], size=2, replace=False)
    
    # ดึงข้อมูลสถิติของ Pokémon ทั้งสองตัว
    p1_data = pokemon_data[pokemon_data["Name"] == p1].iloc[0]
    p2_data = pokemon_data[pokemon_data["Name"] == p2].iloc[0]
    
    # คำนวณ Type Advantage โดยใช้ข้อมูลใน type_effectiveness
    p1_type_1, p1_type_2 = p1_data["Type 1"], p1_data["Type 2"]
    p2_type_1, p2_type_2 = p2_data["Type 1"], p2_data["Type 2"]
    
    # คำนวณการโจมตีจากประเภท
    effectiveness_p1 = 1
    if p2_type_1 in type_effectiveness.get(p1_type_1, {}):
        effectiveness_p1 *= type_effectiveness[p1_type_1][p2_type_1]
    if p2_type_2 in type_effectiveness.get(p1_type_1, {}):
        effectiveness_p1 *= type_effectiveness[p1_type_1][p2_type_2]
    if p2_type_1 in type_effectiveness.get(p1_type_2, {}):
        effectiveness_p1 *= type_effectiveness[p1_type_2][p2_type_1]
    if p2_type_2 in type_effectiveness.get(p1_type_2, {}):
        effectiveness_p1 *= type_effectiveness[p1_type_2][p2_type_2]

    # คำนวณค่า Stat รวม และ Speed
    total_p1, total_p2 = p1_data["Total"], p2_data["Total"]
    speed_p1, speed_p2 = p1_data["Speed"], p2_data["Speed"]

    # เงื่อนไขว่า Total มากกว่า 2 เท่าให้ชนะเลย
    if total_p1 >= 1.3 * total_p2:
        winner = p1
    elif total_p2 >= 1.3 * total_p1:
        winner = p2
    else:
        # คำนวณ Speed Advantage (ให้ Speed สูงกว่าจะโจมตีก่อน)
        if speed_p1 > speed_p2:
            speed_advantage = 1.0  # P1 โจมตีก่อน
        elif speed_p1 < speed_p2:
            speed_advantage = 0.0  # P2 โจมตีก่อน
        else:
            speed_advantage = 0.5  # เสมอกัน

        # คำนวณโอกาสชนะ (Pokémon ที่มี Stat สูงกว่ามีโอกาสชนะมากขึ้น)
        win_probability = (total_p1 * effectiveness_p1 * speed_advantage) / (total_p1 * effectiveness_p1 * speed_advantage + total_p2)
        
        # สุ่มเลือกผู้ชนะตามโอกาสชนะที่คำนวณได้
        winner = p1 if np.random.rand() < win_probability else p2

    battle_data.append([p1, p2, winner])

# สร้าง DataFrame
battle_df = pd.DataFrame(battle_data, columns=["Pokemon_1", "Pokemon_2", "Winner"])

# บันทึกเป็น CSV
battle_df.to_csv("pokemon_battles.csv", index=False)

print("✅ สร้างไฟล์ pokemon_battles.csv สำเร็จ!")
