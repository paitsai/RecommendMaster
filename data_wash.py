def insert_line(file_name, content):
    with open(file_name, 'r', errors='ignore') as file:
        lines = file.readlines()  # 读取所有行
    
    new_lines = []  # 用来存储修改后的行
    
    itmp=1
    for idx in range(len(lines)):
        tmp = lines[idx]
        # 如果当前行不以行号（即idx+1）开头，插入新的行
        if not tmp.startswith(str(idx + itmp)):
            new_lines.append(f"{idx + itmp}::{content}::{content}\n")  # 插入行号加内容
            itmp+=1
        new_lines.append(tmp)  
    
    # 处理完所有行后，写回文件
    with open("./datasets/ml-1m/movies.new.dat", 'w', errors='ignore') as file:
        file.writelines(new_lines)  # 写回所有修改后的行



    with open(file_name, 'w') as file:
        file.writelines(lines)  # 写回所有修改后的行

# 示例
file_name = './datasets/ml-1m/movies.dat'
content = "++++++++++"

insert_line(file_name,content)