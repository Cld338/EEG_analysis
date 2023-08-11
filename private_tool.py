from time import time
from urllib import request
import datetime
import json
import sys
import os

def loadJson(dir :str):
    """Json 파일에서 데이터 로드

    Args:
        dir (str): 파일 경로

    Returns:
        Any: Json 파일에 저장된 데이터
    """
    with open(dir, 'r') as file:
        ls = json.load(file)
    return ls

def saveJson(dir :str, data) -> None:
    """데이터를 Json 형태로 저장

    Args:
        dir (str): 파일 경로
        data (_type_): 데이터
    """
    with open(dir, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent="\t")

def createDirectory(directory :str) -> None:
    """디렉토리 생성

    Args:
        directory (str): 생성하고자 하는 폴더 경로
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def workingDirectory() -> str:
    """현재 디렉토리 반환
    exe 및 py 파일 지원

    Returns:
        str: 경로
    """
    if getattr(sys, 'frozen', False):
        currDir = os.path.dirname(os.path.abspath(sys.executable))    
    else:
        currDir = os.path.dirname(os.path.abspath(__file__))
    return currDir

def filesInFolder(dir :str, extention :str or bool=False) -> list[str]:
    """경로 입력시 해당 폴더 내의 파일 및 폴더 반환

    Args:
        dir (str): 경로
        extention (str or bool, optional): 확장자. Defaults to False.

    Returns:
        list[str]: 폴더 내의 파일 및 폴더 명
    """
    if extention:
        return [file.replace(f".{extention}", "") for file in os.listdir(dir) if file.endswith(f".{extention}")]
    else:
        return [file for file in os.listdir(dir)]

def download_file(url :str, filename :str) -> None:
    """url을 통해 파일 저장

    Args:
        url (str): 파일 url
        filename (str): 저장될 파일명
    """
    try:
        request.urlretrieve(url, filename)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def log(text :str) -> None:
    """현재 디렉토리에 log.txt 파일 생성 및 로그 작성
    현재 시간 - text

    Args:
        text (str): 로그
    """
    currDir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{currDir}/log.txt', 'a') as file:
        file.write(f"{datetime.datetime.now()} - {text}\n")

def checkTime(func, *params) -> float:
    """함수 실행에 걸리는 시간 측정

    Args:
        func (_type_): 함수
        params (list): 함수 실행에 필요한 파라미터

    Returns:
        float: _description_
    """
    start = time()
    output = func(*params)
    return (time()-start, output)

def parentDirectory(dir :str, separator :str="\\", n :int=1) -> str:
    """입력된 디렉토리의 parent 반환

    Args:
        dir (str): 경로
        separator (str): 구분자. Defaults to "\".
        n (int): parent 까지의 차수. Defaults to 1.

    Returns:
        str: 부모 노드의 경로
    """
    return separator.join(dir.split(separator)[:-n])

def writeRequirements(dir_to_code:str) -> None:
    """현재 디렉토리에 requirements.txt 작성

    Args:
        dir_to_code (str): py 파일이 저장된 디렉토리
    """
    libs = set()
    dirs = filesInFolder(dir_to_code, "py")
    for directory in dirs:
        with open(dir_to_code + "/" + directory + ".py", "r", encoding="utf-8") as file:
            txt = file.readlines()
            for line in txt:
                if "import" in line:
                    lib = line.split()[1]
                    if "." in lib:
                        lib = lib.split(".")[0]
                    libs.add(lib.lstrip().rstrip())
    ls = [[i, os.popen(f"pip show {i}").readlines()] for i in libs]
    versions = [f"{i[0]}=={j.split()[1]}" for i in ls for j in i[1] if "Version" in j]
    with open(currDir + "/requirements.txt", "w", encoding="utf-8") as file:
        for i in versions:
            file.write(i+"\n")

currDir = os.path.dirname(os.path.realpath(__file__))