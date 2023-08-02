from time import time
from urllib import request
import datetime
import json
import sys
import os

def loadJson(dir :str) -> list:
    """Json 파일에 저장된 데이터 불러오기"""
    with open(dir, 'r') as file:
        ls = json.load(file)
    return ls

def saveJson(dir :str, data) -> None:
    """데이터를 Json 파일에 저장하기"""
    with open(dir, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent="\t")

def createDirectory(directory :str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def workingDirectory() -> str:
    if getattr(sys, 'frozen', False):
        #test.exe로 실행한 경우,test.exe를 보관한 디렉토리의 full path를 취득
        currDir = os.path.dirname(os.path.abspath(sys.executable))    
    else:
        #python test.py로 실행한 경우,test.py를 보관한 디렉토리의 full path를 취득
        currDir = os.path.dirname(os.path.abspath(__file__))
    return currDir

def filesInFolder(dir :str, extention=False) -> list[str]:
    if extention:
        return [file.replace(f".{extention}", "") for file in os.listdir(dir) if file.endswith(f".{extention}")]
    else:
        return [file for file in os.listdir(dir)]

def download_file(url :str, filename :str) -> None:
    try:
        request.urlretrieve(url, filename)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def log(text :str) -> None:
    currDir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{currDir}/log.txt', 'a') as file:
        file.write(f"{datetime.datetime.now()} - {text}\n")

def inputType(func) -> dict:
    return func.__annotations__

def checkTime(func, *params) -> float:
    start = time()
    output = func(*params)
    return (time()-start, output)

def parentDirectory(dir :str, separator :str="\\", n :int=1) -> str:
    return separator.join(dir.split(separator)[:-n])

currDir = os.path.dirname(os.path.realpath(__file__))