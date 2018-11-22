import csv
import pymongo

mongo_url = "140.115.54.219:27017"

client = pymongo.MongoClient(mongo_url)
DATABASE = "WEDM"
db = client[DATABASE]
COLLECTION = "NCU"
db_coll = db[COLLECTION]

fieldList = [
        '系統時間', 'GAP電壓', '機台進幾率', '放電時間比', '水阻值', '目前加工主程式', '目前加工程式',
        '目前加工行', '目前已放電時間', '預計完成時間', '顯示錯誤號碼',
        '機械座標X', '機械座標Y', '機械座標Z', '機械座標U', '機械座標V',
        '程式座標X', '程式座標Y', '程式座標Z', '程式座標U', '程式座標V',
        'G92座標X', 'G92座標Y', 'G92座標Z', 'G92座標U', 'G92座標V',
        '區域座標X', '區域座標Y', '區域座標Z', '區域座0標U', '區域座標V',
        '當前訊息字串', '放電碼', 'OV', 'LP', 'ON', 'OFF', 'AN', 'AF', 'SV',
        'FR', 'WF', 'WT', 'FL', 'FM', 'FMAX', 'OFT_0', 'OFT_1', 'OFT_2', 'OFT_3', 'RM',
        '下伸臂溫度', '機體溫度', '室內溫度', '冷卻機溫度', '目標溫度',
        '警告號碼', '錯誤號碼', '機台動作', '機台狀態', '機台錯誤訊息',
        '主視窗位置', '子視窗位置', '子視窗功能鍵', '當前N碼', '機台閒置時間',
        'LOCK座標', '旋轉角度',
        'G54座標X', 'G54座標Y', 'G54座標Z', 'G54座標U', 'G54座標V',
        'G55座標X', 'G55座標Y', 'G55座標Z', 'G55座標U', 'G55座標V',
        'G56座標X', 'G56座標Y', 'G56座標Z', 'G56座標U', 'G56座標V',
        'G57座標X', 'G57座標Y', 'G57座標Z', 'G57座標U', 'G57座標V',
        'G58座標X', 'G58座標Y', 'G58座標Z', 'G58座標U', 'G58座標V',
        'G59座標X', 'G59座標Y', 'G59座標Z', 'G59座標U', 'G59座標V',
        '程式總加工長度', '程式剩餘長度', '區段狀態', '系統狀態', '放電開始時數',
        '加工時速度', '控制器微秒時間', '放電功率', '水箱功率', '樹質每秒效率',
        'POWER_METER電壓', 'POWER_METER電流', 'NSCount', 'ASCount', 'SSCount',
        ]
searchRes = db_coll.find()

with open('WEDM.csv', 'w+', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(fieldList)
    for record in searchRes:
        print(f"record = {record}")
        recordValueList = []
        for field in fieldList:
            if field not in record:
                recordValueList.append("None")
            else:
                recordValueList.append(record[field])
        csv_writer.writerow(recordValueList)
