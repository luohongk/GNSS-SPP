"""
Author: Hongkun Luo
Date: 2024-04-07 20:04:05
LastEditors: Hongkun Luo
Description: 

Hongkun Luo
"""

from satelite import Satelite
import numpy as np


class ReadFile:
    # 定义常量成员

    #  粗略的测站坐标
    ApproxPos = [None] * 3

    #  N文件所有行的数据
    NLines = None

    #  O文件所有行的数据
    OLines = None

    # O文件"END OF HEADER"所在的行
    OHeaderLastLine = 0

    # O文件观测类型列表，例如 ['L1','L2','P1','P2','C1','C2','S1','S2']
    ObsTypes = []

    # Klobuchar 电离层改正参数（来自 N 文件头 ION ALPHA / ION BETA）
    IonAlpha = [0.0, 0.0, 0.0, 0.0]
    IonBeta  = [0.0, 0.0, 0.0, 0.0]

    # 类的初始化函数
    def __init__(self, File):

        #  文件路径
        self.oFilePath = File[0]
        self.nFilePath = File[1]

        self.OLines_ = self.ReadOFile()
        self.NLines_ = self.ReadNFile()
        self.ApproxPos = []
        self.ObsTypes = []
        #  O,N文件预处理模块
        self.NHeaderLastLine = self.PreprocessNFile(self.NLines_)
        self.OHeaderLastLine = ReadFile.PreprocessOFile(self, self.OLines_)

        #  常量初始化
        if ReadFile.NLines == None:
            ReadFile.NLines = self.NLines_
        # Bug修复4: 原代码将 OLines 的值错误地赋给了 NLines，导致 OLines 类变量永远为 None
        # Position 类中 self.Lines = ReadFile.OLines 会得到 None，程序直接崩溃
        if ReadFile.OLines == None:
            ReadFile.OLines = self.OLines_
        if ReadFile.ApproxPos == [None] * 3:
            ReadFile.ApproxPos = self.ApproxPos

        if ReadFile.OHeaderLastLine == 0:
            ReadFile.OHeaderLastLine = self.OHeaderLastLine

        if not ReadFile.ObsTypes:
            ReadFile.ObsTypes = self.ObsTypes

        # 传播 Klobuchar 电离层参数到类变量（仅首次实例化时赋值）
        if ReadFile.IonAlpha == [0.0, 0.0, 0.0, 0.0]:
            ReadFile.IonAlpha = self._ion_alpha
        if ReadFile.IonBeta == [0.0, 0.0, 0.0, 0.0]:
            ReadFile.IonBeta = self._ion_beta

        self.Satelites = []

        # 这个表示卫星的位置，初始化的时候是
        self.Pos = []

        self.PosName = []

        self.Time = []

        self.RefTime = []

        # 钟飘？反正传入的时候传一个含有三个元素的数组就可以
        self.SateliteClockCorrect = []

        # 这个是卫星观测值，用于计算卫星的位置，本项目中是一个6乘4的矩阵
        self.SateliteObservation = []

    # Bug修复5: 以下 getter 方法用 @classmethod 装饰但访问实例变量（如 NHeaderLastLine），
    # 会导致 AttributeError。访问类变量的方法改为真正的 classmethod（参数改为 cls），
    # 访问实例变量的方法去掉 @classmethod 改为普通实例方法。
    @classmethod
    def GetApproxPos(cls):
        return cls.ApproxPos

    @classmethod
    def GetOLines(cls):
        return cls.OLines

    @classmethod
    def GetOHeaderLastLine(cls):
        return cls.OHeaderLastLine

    @classmethod
    def GetNlines(cls):
        return cls.NLines

    def GetNHeaderLastLine(self):
        return self.NHeaderLastLine

    @classmethod
    def GetObsTypes(cls):
        return cls.ObsTypes

    def ReadNFile(self):
        with open(self.nFilePath, "r") as file:
            lines = file.readlines()
            ReadFile.NLines = lines
            return lines

    def ReadOFile(self):
        with open(self.oFilePath, "r") as file:
            lines = file.readlines()
            ReadFile.OLines = lines
            return lines

    def PreprocessNFile(self, lines):
        """解析 N 文件头：提取 END OF HEADER 行号和 Klobuchar 电离层参数。"""

        def _parse_rinex_float(s):
            """将 RINEX/Fortran D-notation 字符串转换为 float，如 '2.9802D-08'。"""
            return float(s.strip().replace('D', 'E').replace('d', 'e'))

        # 寻找END OF HEADER所在的行
        target_string = "END OF HEADER"
        HeadLine = 0
        ion_alpha = [0.0, 0.0, 0.0, 0.0]
        ion_beta  = [0.0, 0.0, 0.0, 0.0]

        # 遍历行，如果找到了end of header就记录并且退出
        for i, line in enumerate(lines, start=1):
            if "ION ALPHA" in line:
                try:
                    # RINEX 2.11: 4个值，每个12字节，从第2列开始（0-indexed col 2）
                    for k in range(4):
                        col = 2 + k * 12
                        ion_alpha[k] = _parse_rinex_float(line[col:col+12])
                except Exception:
                    pass
            if "ION BETA" in line:
                try:
                    for k in range(4):
                        col = 2 + k * 12
                        ion_beta[k] = _parse_rinex_float(line[col:col+12])
                except Exception:
                    pass
            if target_string in line:
                HeadLine = i
                break

        # 将解析结果存入实例属性，供 __init__ 传递到类变量
        self._ion_alpha = ion_alpha
        self._ion_beta  = ion_beta
        return HeadLine
        # 此处是拓展部分,由于我已经知道我的文件是GPS数据了,就直接命名卫星名字为GXXX
        # 这里可以自己拓展一下
        # if('GPS' in lines[0]):
        #     obs_type='GPS'

    # O文件的预处理，进行粗略坐标和观测类型的读取
    def PreprocessOFile(self, lines):
        ApproxPosComment = "APPROX POSITION XYZ"
        ObsTypeComment = "# / TYPES OF OBSERV"

        for i, line in enumerate(lines, start=1):
            # 读取粗略坐标
            if ApproxPosComment in line:
                approx_x = float(line[0:15].strip())
                approx_y = float(line[15:28].strip())
                approx_z = float(line[29:42].strip())
                self.ApproxPos.append(approx_x)
                self.ApproxPos.append(approx_y)
                self.ApproxPos.append(approx_z)

            # 读取观测类型列表（RINEX 2.11）
            # 首行: col[0:6] 为类型总数（数字）；续行: col[0:6] 为空格
            # 两种行的 col[0:6] 可用 isdigit() 区分
            # 无论首行还是续行，类型名均从 col[6] 起每6字节一个
            if ObsTypeComment in line:
                for k in range(9):
                    start = 6 + k * 6
                    if start >= 60:
                        break
                    obs = line[start:start + 6].strip()
                    if obs:
                        self.ObsTypes.append(obs)

        ObsTargetString = "END OF HEADER"
        ObsHeaderLine = 0

        # 寻找END OF HEADER所在的行
        for i, line in enumerate(lines, start=1):
            if ObsTargetString in line:
                ObsHeaderLine = i
                break

        return ObsHeaderLine

    def CaculateSatRefTime(Time):
        return 0

    def CaculateSatelites(self):
        for i in range(self.NHeaderLastLine, len(self.NLines) - 9, 8):

            line = self.NLines[i]
            num = line[0:2].strip()

            time = [None] * 6
            # print(line[i+2])
            time[0] = int((line[3:5]).strip()) + 2000
            time[1] = int((line[6:8]).strip())
            time[2] = int((line[9:11]).strip())
            time[3] = int((line[12:14]).strip())
            time[4] = int((line[15:17]).strip())
            time[5] = float((line[18:22]).strip())

            # 读取卫星钟差改正参数
            time_change = []
            a = float(line[22:37].strip()) * pow(10, int(line[38:41].strip()))
            time_change.append(a)

            b = float(line[41:56].strip()) * pow(10, int(line[57:60].strip()))
            time_change.append(b)

            c = float(line[60:75].strip()) * pow(10, int(line[76:79].strip()))
            time_change.append(c)

            self.SateliteClockCorrect.append(time_change)

            rows = 6
            cols = 4
            matrix = np.zeros((rows, cols))

            # 读取卫星位置计算的参数
            for j in range(0, rows):
                for k in range(0, cols):
                    matrix[j][k] = float(
                        self.NLines[i + 1 + j][3 + 19 * k : 18 + 19 * k]
                    ) * pow(10, int(self.NLines[i + 1 + j][19 + 19 * k : 22 + 19 * k]))

            self.SateliteObservation.append(matrix)

            SateliteName = f"G{int(num):02d}"
            # SateliteRefTime=ReadFile.CaculateSatRefTime(time)

            satelite = Satelite(SateliteName, time, time_change, matrix)

            # 这里需要传入观测时间，计算卫星坐标
            satelite.InitPositionOfSat(time)

            self.Pos.append([satelite.X, satelite.Y, satelite.Z])
            self.Time.append(time)
            self.PosName.append(SateliteName)
            # self.RefTime.append(SateliteRefTime)

            self.Satelites.append(satelite)
