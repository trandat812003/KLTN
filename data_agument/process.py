import json

def load_json_from_text_file(txt_file_path):
    """
    Đọc từng dòng trong tệp txt và coi mỗi dòng là một JSON, sau đó trả về danh sách các JSON từ tệp txt.
    """
    json_data_list = []
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
        for line in txt_file:
            try:
                json_data = json.loads(line)
                json_data_list.append(json_data)
            except json.JSONDecodeError:
                print("Không thể đọc JSON từ dòng:", line.strip())
    return json_data_list


def process_files(txt_file_path, json_file_path, output_txt_file_path):
    """
    Đọc tệp txt và json, kiểm tra sự tồn tại của 'situation' trong json.
    Nếu có, lấy 'survey_score' và thêm vào dữ liệu txt.
    """
    # Load dữ liệu từ tệp JSON
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Load các JSON từ tệp txt
    txt_json_list = load_json_from_text_file(txt_file_path)

    # Duyệt qua từng JSON trong tệp txt và xử lý
    with open(output_txt_file_path, 'w', encoding='utf-8') as output_txt_file:
        for txt_json in txt_json_list:
            # Lấy giá trị của 'situation'
            situation_key = txt_json.get('situation')
            if situation_key is None:
                print("Không tìm thấy 'situation' trong JSON:", txt_json)
            else:
                # Tìm 'situation_key' trong json_data để lấy 'survey_score'
                matching_situation = next((item for item in json_data if item.get('situation') == situation_key), None)
                if matching_situation:
                    txt_json['survey_score'] = matching_situation.get('survey_score', '')
                    txt_json['seeker_question1'] = matching_situation.get('seeker_question1', '')
                    txt_json['seeker_question2'] = matching_situation.get('seeker_question2', '')
                    txt_json['supporter_question1'] = matching_situation.get('supporter_question1', '')
                    txt_json['supporter_question2'] = matching_situation.get('supporter_question2', '')
                    txt_json['persona'] = matching_situation.get('persona', '')
                    txt_json['persona_list'] = matching_situation.get('persona_list', '')
                else:
                    print(f"Không tìm thấy tình huống '{situation_key}' trong tệp JSON")
                
                # Ghi lại JSON đã cập nhật vào tệp txt đầu ra
                output_txt_file.write(json.dumps(txt_json) + '\n')


# Đường dẫn tệp
txt_file_path = '/home/trandat/Documents/lab/KLTN/dataset/esconv/sbert/valid.txt'
json_file_path = '/home/trandat/Documents/lab/KLTN/data_agument/PESConv.json'
output_txt_file_path = '/home/trandat/Documents/lab/KLTN/dataset/esconv/pesconv/valid.txt' 

# Gọi hàm xử lý
process_files(txt_file_path, json_file_path, output_txt_file_path)
