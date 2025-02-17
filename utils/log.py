import utils.globalvar as gl
import os 
import os.path as osp
import datetime

LOG_LEVELS = {
    'stat': 3,    # stat 只输出统计信息，用来分析
    'warn': 2,    # warn 和info统一级别
    'info': 2,    # info 级别输出 info 和 detail
    'detail': 1   # detail 级别只输出 detail
}

LEVEL_HIERARCHY = ['stat', 'warn', 'info', 'detail']

def should_log(current_level, log_level):
    loggingInfo=gl.get_value("LoggingInfo")
    log_level=loggingInfo["level"]

    current_priority = LOG_LEVELS.get(current_level, 3)
    log_priority = LOG_LEVELS.get(log_level, 3)

    return log_priority <= current_priority

def initLoggingConfig(level="detail",logFileName="",logPath=""):
    loggingInfo={}
    loggingInfo["level"]=level
    loggingInfo["logFileName"]=initFileName(logFileName)
    loggingInfo["logPath"]=logPath
    gl.set_value("LoggingInfo",loggingInfo)
    
    if not os.path.exists(loggingInfo["logPath"]):
        os.makedirs(loggingInfo["logPath"])  # 如果目录不存在，创建目录

def initFileName(fileName):
    fileSuffix = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{fileName}_{fileSuffix}"


def getLoggingFilePath(level):
    loggingInfo=gl.get_value("LoggingInfo")
    fileName=loggingInfo["logFileName"]
    log_filename = f"{level}_{fileName}.log"
    return osp.join(loggingInfo["logPath"],log_filename)

# 分级控制的输出文件
def loggingToFile(logMessage,level):    
    for log in LEVEL_HIERARCHY:
        if LOG_LEVELS[log] <= LOG_LEVELS[level] and should_log(level, log):
            logFilePath = getLoggingFilePath(log)
            with open(logFilePath, 'a') as logFile:
                logFile.write(logMessage)

def processMessage(message, level):
    logTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logMessage = f"[{logTime}] [{level.upper()}] "
    
    # 将消息按行分割
    lines = message.split('\n')
    
    # 第一行不缩进，后续行缩进
    logMessage += lines[0]  # 第一行直接加
    
    # 处理第二行及之后的行
    if len(lines) > 1:
        for line in lines[1:]:  # 后续行缩进
            logMessage += f"\n\t\t\t\t\t\t{line}"

    logMessage += "\n"  # 添加换行符以结束
    return logMessage

#log 的实际过程
def logging(message,toConsole, level):
    #预处理
    logMessage = processMessage(message,level)
    #输出到分级文件中，这里会进行分级控制
    loggingToFile(logMessage,level)
    #输出到控制台
    if toConsole:
        print(logMessage)

def loggingInfo(message, toConsole=True):
    logging(message, toConsole, "info")

def loggingDetail(message, toConsole=False):
    logging(message, toConsole, "detail")

def loggingWarn(message, toConsole=True):
    logging(message, toConsole, "warn")

def loggingStat(message, toConsole=True):
    logging(message, toConsole, "stat")
    
if __name__ == '__main__':
    gl._init()
    initLoggingConfig(level="detail",logFileName='{}_Mlti_2{}_v1_2'.format("dataSet",'every'),logPath="logs")
    loggingDetail("this is detail")
    loggingInfo("this is info")
    loggingWarn("this is warn")
    loggingStat("this is stat")