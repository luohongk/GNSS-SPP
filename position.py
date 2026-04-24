"""
Author: Hongkun Luo
Date: 2024-04-07 20:43:56
LastEditors: Hongkun Luo
Description: 

Hongkun Luo
"""

from readfile import ReadFile
from satelite import Satelite
import datetime
import numpy as np


class Position:

    def __init__(self, SateliteObservation, SateliteName, Time, SateliteClockCorrect):

        # 卫星索引列表
        self.SateliteObservation = SateliteObservation
        self.SateliteName = SateliteName
        self.Time = Time
        self.SateliteClockCorrect = SateliteClockCorrect

        # 获取O文件的数据，这就包括文本数据以及END OF Header所在的行
        self.Lines = ReadFile.OLines
        self.OHeaderLastLine = ReadFile.OHeaderLastLine

        # 获取坐标粗略值
        self.ApproxPos = ReadFile.ApproxPos

        # 读取观测类型列表，动态计算每颗卫星占用的行数以及P1/C1列偏移
        # RINEX 2.11: 每行最多5个观测值（abpo为8型每星2行；al2h为14型每星3行）
        obs_types = ReadFile.ObsTypes
        num_obs = len(obs_types)
        # 每颗卫星占用的数据行数 = ceil(num_obs / 5)
        import math
        self.LinesPerSat = math.ceil(num_obs / 5) if num_obs > 0 else 2

        # 计算P1在哪一行(row_idx, 0-based)和哪一列(col_byte_offset)
        # 每行最多5个值，每个值16字节(F14.3+LLI+signal)
        self.P1ColRow = -1
        self.P1ColOffset = -1
        self.C1ColRow = -1
        self.C1ColOffset = -1
        for idx, otype in enumerate(obs_types):
            row = idx // 5        # 在第几行（0-based）
            col = idx % 5         # 在该行第几个（0-based）
            byte_start = col * 16  # 字段起始字节
            if otype == 'P1':
                self.P1ColRow = row
                self.P1ColOffset = byte_start
            if otype == 'C1':
                self.C1ColRow = row
                self.C1ColOffset = byte_start

    def GenerateObs(self):
        return 0

    def validate_pseudorange(self, pr, geom_dist, satellite_id=""):
        """
        检验伪距观测值的物理有效性。
        
        伪距方程: PR = ρ + c*dtr - c*dts + delays + noise
        约束条件: 接收机钟差通常在 ±10ms 内，所以:
                PR 应该接近 ρ，误差在几米到几百米之间
        
        Args:
            pr: 观测伪距 (米)
            geom_dist: 几何距离 (米)
            satellite_id: 卫星标识 (用于日志输出)
        
        Returns:
            (is_valid, message)
        """
        c = 299792458
        
        # 允许的最大钟差 (秒) - 典型值为 ±0.01 秒 = ±10 ms
        max_clock_bias = 0.01
        max_bias_meters = max_clock_bias * c
        
        # 物理约束检验
        # 伪距不应该显著小于几何距离
        if pr < geom_dist - max_bias_meters:
            diff_km = (pr - geom_dist) / 1000
            return False, f"Invalid PR for {satellite_id}: {diff_km:.1f} km < expected, implies unrealistic clock bias"
        
        # 伪距也不应该显著大于几何距离
        # (但这个约束较松，因为阴离子延迟和系统误差会使PR更大)
        if pr > geom_dist + 2 * max_bias_meters:
            diff_km = (pr - geom_dist) / 1000
            return False, f"Suspicious PR for {satellite_id}: {diff_km:.1f} km > expected, excessive clock bias"
        
        return True, "Valid"

    def MatchObservationAndCaculate(self):
        # line用于临时存储的行数
        line = self.Lines[self.OHeaderLastLine]

        # 这个变量用于标记读取的行数
        NReadLine = self.OHeaderLastLine

        # 当前近似坐标（随每个历元的解更新，逐渐收敛到真值）
        CurrentApproxPos = list(self.ApproxPos)

        while line != "":
            # 跳过空行
            if line.strip() == "":
                NReadLine += 1
                if NReadLine >= len(self.Lines):
                    break
                line = self.Lines[NReadLine]
                continue

            # RINEX 2.11 历元记录首字节为空格，col[1:3]为年份
            # 如果该行不像历元记录（如文件尾的注释行），跳过
            if line[0] != ' ' or not line[1:3].strip().isdigit():
                NReadLine += 1
                if NReadLine >= len(self.Lines):
                    break
                line = self.Lines[NReadLine]
                continue

            # 这里表示从O文件获取当前观测时间
            obs_time = [None] * 6
            obs_time[0] = int((line[1:3]).strip()) + 2000
            obs_time[1] = int((line[4:6]).strip())
            obs_time[2] = int((line[7:9]).strip())
            obs_time[3] = int((line[10:12]).strip())
            obs_time[4] = int((line[13:15]).strip())
            # Bug修复6: RINEX 2.11 秒数为 F10.7 格式，位于 col[15:26]（共11字节）
            # 原代码 [17:18] 只读了1个字符，导致秒数 >= 10 时解析错误（如 30s 读成 0）
            obs_time[5] = float((line[15:26]).strip())

            # 这里表示当前观测时间下，有多少个卫星观测值
            num_sat = int(line[30:32])

            # 这里希望获取当前观测时间下,卫星名称列表在O文件所占有的行数
            if num_sat % 12 == 0:
                n = int(num_sat / 12)
            else:
                n = int((num_sat / 12)) + 1

            str = ""

            for j in range(n):
                str = str + self.Lines[NReadLine + j][32:68].strip()
            # 这里表示当前观测时间下，卫星的名称列表
            obs_sat_PRN = []
            for k in range(num_sat):
                obs_sat_PRN.append(str[k * 3 : k * 3 + 3])

            NReadLine = NReadLine + n

            # 接下来获取伪距列表（动态行数和列偏移，兼容任意RINEX 2.11观测文件）
            ObsPseudorange = []
            lps = self.LinesPerSat  # 每颗卫星占用的行数
            for j in range(NReadLine, NReadLine + lps * num_sat, lps):
                # P1: 第 P1ColRow 行，列偏移 P1ColOffset，字段宽14字节
                Pseudorange = 0
                if self.P1ColRow >= 0:
                    p1_line = self.Lines[j + self.P1ColRow]
                    p1_start = self.P1ColOffset
                    p1_str = p1_line[p1_start:p1_start + 14].strip() if len(p1_line) > p1_start + 14 else ""
                    if p1_str:
                        Pseudorange = float(p1_str)

                # C1 备选：P1为空时使用C1
                if Pseudorange == 0 and self.C1ColRow >= 0:
                    c1_line = self.Lines[j + self.C1ColRow]
                    c1_start = self.C1ColOffset
                    c1_str = c1_line[c1_start:c1_start + 14].strip() if len(c1_line) > c1_start + 14 else ""
                    if c1_str:
                        Pseudorange = float(c1_str)

                ObsPseudorange.append(Pseudorange)

            # 直接计算当前位置下的测站位置并且打印
            SatLiteXYZ = self.MatchToSatlite(obs_time, obs_sat_PRN)

            # print(ObsPseudorange)
            # print(SatLiteXYZ)

            # 进行非线性最小二乘，平差计算地面坐标
            result = self.SolutionLeastSquares(
                ObsPseudorange, SatLiteXYZ, CurrentApproxPos
            )
            # 用本历元解更新近似坐标，供下一历元使用
            if isinstance(result, list):
                CurrentApproxPos = result

            # 更新读取的行数
            NReadLine = NReadLine + lps * num_sat

            # Bug修复9: 添加数组越界检查，防止读取到文件末尾时 IndexError
            if NReadLine >= len(self.Lines):
                break

            # 更新读取的内容
            line = self.Lines[NReadLine]

    def MatchToSatlite(self, ObsTime, ObsSatPrn):

        # 匹配结果保存

        SatLiteXYZ = []
        # 进行卫星匹配
        # 遍历观测值文件某个时间下的卫星编号
        # AfterMatch
        for index, SatPRN in enumerate(ObsSatPrn):
            # 遍历卫星参考时刻的PRN号

            #   这四个元素的说明，前三个元素保存卫星坐标，后一个保存是否匹配上的一个标签值,卫星钟差
            TemXYZ = [None] * 5

            TimeDiff = []
            for index1, SatPRN1 in enumerate(self.SateliteName):
                if SatPRN == SatPRN1:
                    # 计算时间差，取绝对值
                    # Bug修复8: 原代码未取绝对值，当观测时刻晚于参考时刻时（时间差为负），
                    # min() 会选出负值最大的（即差最远的星历），而非最近的
                    TimeDiff.append(
                        abs(self.CaculateTimeDifference(ObsTime, self.Time[index1]))
                    )
                else:
                    # 这里设置三天所有的秒数，保证足够大，找最小值的之后不找到就可以了
                    TimeDiff.append(2592000)

            #     没有匹配上的判断
            NotMatch = all(x == 2592000 for x in TimeDiff)
            if NotMatch == True:
                TemXYZ[0] = 0
                TemXYZ[1] = 0
                TemXYZ[2] = 0
                TemXYZ[3] = 0
                TemXYZ[4] = 0
            else:
                # 寻找最小时间差的索引
                MinTime = min(TimeDiff)
                MinTimeindex = TimeDiff.index(MinTime)

                # 计算这个最小索引卫星的坐标
                satelite = Satelite(
                    self.SateliteName[MinTimeindex],
                    self.Time[MinTimeindex],
                    self.SateliteClockCorrect[MinTimeindex],
                    self.SateliteObservation[MinTimeindex],
                )

                # 这里需要传入观测时间，计算卫星坐标
                satelite.InitPositionOfSat(ObsTime)
                TemXYZ[0] = satelite.X
                TemXYZ[1] = satelite.Y
                TemXYZ[2] = satelite.Z
                TemXYZ[3] = 1
                TemXYZ[4] = satelite.Delta_T
            SatLiteXYZ.append(TemXYZ)
        return SatLiteXYZ

        # 计算伪距所对应的卫星的位置

    def CaculateTimeDifference(self, Time1, Time2):
        time1 = datetime.datetime(
            Time1[0], Time1[1], Time1[2], Time1[3], Time1[4], int(Time1[5])
        )
        time2 = datetime.datetime(
            Time2[0], Time2[1], Time2[2], Time2[3], Time2[4], int(Time2[5])
        )
        diff = time2 - time1
        seconds = diff.total_seconds()
        return seconds

    # 最小二乘算法求解（带迭代收敛）
    def SolutionLeastSquares(self, ObsPseudorange, SatLiteXYZ, ApproxPos):
        # 判断匹配上的伪距是否多于四个，因为最小二乘至少要四个观测方程，求解地面坐标，求解接收机钟差

        SizeSatLiteXYZ = len(SatLiteXYZ)
        c = 299792458  # 光速（精确值 m/s）

        # 统计有效观测数（已匹配且伪距非零）
        valid_count = sum(
            1 for k in range(SizeSatLiteXYZ)
            if SatLiteXYZ[k][3] == 1 and ObsPseudorange[k] != 0
        )
        if valid_count < 4:
            print("无法进行平差计算（有效卫星数不足4颗）")
            return ApproxPos

        # 迭代最小二乘，最多迭代10次，收敛阈值0.001m
        CurrentPos = list(ApproxPos)
        dtr = 0.0  # 接收机钟差（单位：秒）
        for iteration in range(10):
            B = []
            L = []

            for k in range(SizeSatLiteXYZ):
                # 跳过未匹配的卫星
                if SatLiteXYZ[k][3] == 0:
                    continue
                # 跳过无效伪距（P1和C1均为空）
                if ObsPseudorange[k] == 0:
                    continue

                # 计算当前近似位置到卫星的几何距离
                P0 = np.sqrt(
                    np.square(SatLiteXYZ[k][0] - CurrentPos[0])
                    + np.square(SatLiteXYZ[k][1] - CurrentPos[1])
                    + np.square(SatLiteXYZ[k][2] - CurrentPos[2])
                )

                # 设计矩阵B每一行：方向余弦 + 接收机钟差系数
                TemB0 = -1 * (SatLiteXYZ[k][0] - CurrentPos[0]) / P0
                TemB1 = -1 * (SatLiteXYZ[k][1] - CurrentPos[1]) / P0
                TemB2 = -1 * (SatLiteXYZ[k][2] - CurrentPos[2]) / P0
                TemB3 = 1.0
                B.append([TemB0, TemB1, TemB2, TemB3])

                # 常数项L：观测伪距 − 几何距离 + 卫星钟差改正
                # 伪距观测方程: P = ρ + c*dtr - c*dts
                # 改写: P - ρ + c*dts = c*dtr + (ρ真 - ρ近似)的线性化部分
                TemLRol = ObsPseudorange[k] - P0 + c * SatLiteXYZ[k][4]
                L.append(TemLRol)

            ArrayB = np.array(B)
            ArrayL = np.array(L)

            # 间接平差公式: x = (B^T B)^-1 B^T L
            BtB = np.transpose(ArrayB) @ ArrayB
            # 检查矩阵条件数，避免病态矩阵
            if np.linalg.matrix_rank(BtB) < 4:
                print("设计矩阵秩不足，无法求解")
                return ApproxPos

            x = np.linalg.inv(BtB) @ (np.transpose(ArrayB) @ ArrayL)

            # 更新近似坐标
            CurrentPos[0] += x[0]
            CurrentPos[1] += x[1]
            CurrentPos[2] += x[2]
            dtr += x[3] / c

            # 收敛判断：坐标改正量小于0.11mm则停止迭代
            if np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < 0.001:
                break

        print(
            f"平差后的X坐标: {CurrentPos[0]:.3f}",
            f"平差后的Y坐标: {CurrentPos[1]:.3f}",
            f"平差后的Z坐标: {CurrentPos[2]:.3f}",
            f"(迭代次数: {iteration+1})",
        )

        return CurrentPos
