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

    # ------------------------------------------------------------------ #
    #  辅助工具：卫星高度角与方位角
    # ------------------------------------------------------------------ #
    @staticmethod
    def calc_elevation_azimuth(rec_xyz, sat_xyz):
        """
        计算卫星相对于接收机的高度角和方位角。

        Args:
            rec_xyz: 接收机 ECEF 坐标 [X, Y, Z]（米）
            sat_xyz: 卫星   ECEF 坐标 [X, Y, Z]（米）

        Returns:
            elev_deg: 高度角（度，-90~90）
            azim_deg: 方位角（度，0~360，北为0顺时针）
        """
        dx = sat_xyz[0] - rec_xyz[0]
        dy = sat_xyz[1] - rec_xyz[1]
        dz = sat_xyz[2] - rec_xyz[2]

        # 接收机大地纬度 φ 和经度 λ
        r_xy = np.sqrt(rec_xyz[0]**2 + rec_xyz[1]**2)
        lat  = np.arctan2(rec_xyz[2], r_xy)   # 地心纬度（近似）
        lon  = np.arctan2(rec_xyz[1], rec_xyz[0])

        # 旋转矩阵：ECEF → 站心坐标系（ENU）
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)

        # ENU = R * d
        e =  -sin_lon * dx + cos_lon * dy
        n =  -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u =   cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        hor  = np.sqrt(e**2 + n**2)
        elev = np.arctan2(u, hor)            # 高度角（rad）
        azim = np.arctan2(e, n)              # 方位角（rad）
        if azim < 0:
            azim += 2 * np.pi

        return np.degrees(elev), np.degrees(azim)

    # ------------------------------------------------------------------ #
    #  Klobuchar 电离层改正（GPS L1，单位：秒，调用时 ×c 得米）
    # ------------------------------------------------------------------ #
    @staticmethod
    def klobuchar_iono(rec_xyz, sat_xyz, obs_time, alpha, beta):
        """
        Klobuchar 电离层时延改正（ICD-GPS-200，L1 频率）。

        Args:
            rec_xyz : 接收机 ECEF [X, Y, Z]（米）
            sat_xyz : 卫星   ECEF [X, Y, Z]（米）
            obs_time: 观测时刻 [year, mon, day, hour, min, sec]
            alpha   : 电离层 α 系数列表（4个，来自 N 文件头）
            beta    : 电离层 β 系数列表（4个，来自 N 文件头）

        Returns:
            dion_sec: 电离层时延（秒）
        """
        elev_deg, azim_deg = Position.calc_elevation_azimuth(rec_xyz, sat_xyz)
        El    = elev_deg / 180.0          # 高度角（半圆）
        Az    = np.radians(azim_deg)

        # 接收机大地纬度（半圆）
        r_xy  = np.sqrt(rec_xyz[0]**2 + rec_xyz[1]**2)
        lat_r = np.arctan2(rec_xyz[2], r_xy)
        phi_u = lat_r / np.pi             # 半圆

        # 地心角（半圆）
        psi = 0.0137 / (El + 0.11) - 0.022

        # 穿刺点纬度（半圆）
        phi_i = phi_u + psi * np.cos(Az)
        phi_i = max(-0.416, min(0.416, phi_i))

        # 穿刺点经度（半圆）
        lon_r = np.arctan2(rec_xyz[1], rec_xyz[0])
        lam_u = lon_r / np.pi
        lam_i = lam_u + psi * np.sin(Az) / np.cos(phi_i * np.pi)

        # 地磁纬度（半圆）
        phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * np.pi)

        # 本地时（秒）
        t_gps  = (obs_time[3] * 3600 + obs_time[4] * 60 + obs_time[5])
        t_loc  = 43200.0 * lam_i + t_gps
        t_loc  = t_loc % 86400.0

        # 振幅 AMP 和 周期 PER
        AMP = sum(alpha[n] * phi_m**n for n in range(4))
        PER = sum(beta[n]  * phi_m**n for n in range(4))
        AMP = max(AMP, 0.0)
        PER = max(PER, 72000.0)

        x = 2.0 * np.pi * (t_loc - 50400.0) / PER

        # 倾斜因子
        F = 1.0 + 16.0 * (0.53 - El)**3

        if abs(x) < 1.57:
            dion_sec = F * (5e-9 + AMP * (1.0 - x**2/2.0 + x**4/24.0))
        else:
            dion_sec = F * 5e-9

        return dion_sec   # 单位：秒（×c 得米）

    # ------------------------------------------------------------------ #
    #  Saastamoinen 对流层改正（天顶方向，简化映射函数）
    # ------------------------------------------------------------------ #
    @staticmethod
    def saastamoinen_tropo(rec_xyz, sat_xyz):
        """
        Saastamoinen 对流层天顶延迟 + 1/sin(elev) 映射函数（精度 ~1m）。

        Args:
            rec_xyz: 接收机 ECEF [X, Y, Z]（米）
            sat_xyz: 卫星   ECEF [X, Y, Z]（米）

        Returns:
            dtrop_m: 对流层距离延迟（米）
        """
        elev_deg, _ = Position.calc_elevation_azimuth(rec_xyz, sat_xyz)
        if elev_deg < 2.0:
            elev_deg = 2.0          # 防止映射函数奇异

        elev = np.radians(elev_deg)

        # 接收机高程（粗略用 |Z| 推算，正式应转到大地坐标）
        r    = np.sqrt(rec_xyz[0]**2 + rec_xyz[1]**2 + rec_xyz[2]**2)
        h    = r - 6371000.0        # 近似高程（米）
        h    = max(h, 0.0)

        # 标准气压 / 温度（海平面修正到站高）
        P    = 1013.25 * (1.0 - 2.2557e-5 * h)**5.2568   # hPa
        T    = 288.15  - 6.5e-3 * h                        # K
        e    = 50.0 * np.exp(-0.0006396 * h)               # hPa 水汽压（RH≈50%）

        # Saastamoinen 天顶延迟（米）
        Ztrop = 0.002277 / np.sin(elev) * (P + (1255.0/T + 0.05) * e
                                            - 4.3e-2 / (np.tan(elev)**2))
        return max(Ztrop, 0.0)

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
                ObsPseudorange, SatLiteXYZ, CurrentApproxPos, obs_time
            )
            # 用本历元解更新近似坐标，供下一历元使用
            if isinstance(result, list):
                CurrentApproxPos = result

            # 更新读取的行数
            NReadLine = NReadLine + lps * num_sat


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
    def SolutionLeastSquares(self, ObsPseudorange, SatLiteXYZ, ApproxPos, ObsTime=None):
        """
        迭代加权最小二乘定位，包含：
          1. 高度角截止角滤波（15°）
          2. Klobuchar 电离层改正
          3. Saastamoinen 对流层改正
          4. 地球自转改正（Sagnac）
          5. 后验残差粗差剔除（阈值 30 m）
        """
        SizeSatLiteXYZ = len(SatLiteXYZ)
        c   = 299792458          # 光速 m/s
        we  = 7.2921151467e-5    # 地球自转角速度 rad/s（WGS-84）
        ELEV_MASK  = 15.0        # 高度角截止角（度）
        GROSS_THR  = 30.0        # 后验残差粗差阈值（米）

        # 读取 Klobuchar 参数（N 文件头解析结果）
        alpha = ReadFile.IonAlpha
        beta  = ReadFile.IonBeta

        # 统计有效观测数（已匹配且伪距非零）
        valid_count = sum(
            1 for k in range(SizeSatLiteXYZ)
            if SatLiteXYZ[k][3] == 1 and ObsPseudorange[k] != 0
        )
        if valid_count < 4:
            print("无法进行平差计算（有效卫星数不足4颗）")
            return ApproxPos

        CurrentPos = list(ApproxPos)
        dtr = 0.0    # 接收机钟差（秒）

        for iteration in range(50):
            B, L, residual_idx = [], [], []

            for k in range(SizeSatLiteXYZ):
                if SatLiteXYZ[k][3] == 0 or ObsPseudorange[k] == 0:
                    continue

                # ------ 1. 高度角截止角 --------------------------------
                elev, _ = Position.calc_elevation_azimuth(
                    CurrentPos,
                    [SatLiteXYZ[k][0], SatLiteXYZ[k][1], SatLiteXYZ[k][2]]
                )
                if elev < ELEV_MASK:
                    continue      # 低仰角卫星直接剔除

                # ------ 2. 地球自转改正（Sagnac）-----------------------
                P0_approx = np.sqrt(
                    (SatLiteXYZ[k][0] - CurrentPos[0])**2
                    + (SatLiteXYZ[k][1] - CurrentPos[1])**2
                    + (SatLiteXYZ[k][2] - CurrentPos[2])**2
                )
                tau   = P0_approx / c
                theta = we * tau
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                sat_x =  SatLiteXYZ[k][0] * cos_t + SatLiteXYZ[k][1] * sin_t
                sat_y = -SatLiteXYZ[k][0] * sin_t + SatLiteXYZ[k][1] * cos_t
                sat_z =  SatLiteXYZ[k][2]

                # 用改正后卫星坐标计算精确几何距离
                P0 = np.sqrt(
                    (sat_x - CurrentPos[0])**2
                    + (sat_y - CurrentPos[1])**2
                    + (sat_z - CurrentPos[2])**2
                )

                # ------ 3. 电离层改正（Klobuchar）----------------------
                if ObsTime is not None and any(a != 0.0 for a in alpha):
                    dion = Position.klobuchar_iono(
                        CurrentPos, [sat_x, sat_y, sat_z],
                        ObsTime, alpha, beta
                    ) * c   # 秒 → 米
                else:
                    dion = 0.0

                # ------ 4. 对流层改正（Saastamoinen）-------------------
                dtrop = Position.saastamoinen_tropo(
                    CurrentPos, [sat_x, sat_y, sat_z]
                )

                # ------ 5. 设计矩阵行 & 常数项 -------------------------
                TemB0 = -(sat_x - CurrentPos[0]) / P0
                TemB1 = -(sat_y - CurrentPos[1]) / P0
                TemB2 = -(sat_z - CurrentPos[2]) / P0
                TemB3 = 1.0
                B.append([TemB0, TemB1, TemB2, TemB3])

                # 伪距方程: P = ρ + c*dtr - c*dts + dion + dtrop
                # 常数项  : l = P - ρ + c*dts - dion - dtrop
                TemL = ObsPseudorange[k] - P0 + c * SatLiteXYZ[k][4] - dion - dtrop
                L.append(TemL)
                residual_idx.append(k)

            if len(B) < 4:
                print("高度角截止后有效卫星不足4颗，跳过本历元")
                return ApproxPos

            ArrayB = np.array(B)
            ArrayL = np.array(L)

            BtB = np.transpose(ArrayB) @ ArrayB
            if np.linalg.matrix_rank(BtB) < 4:
                print("设计矩阵秩不足，无法求解")
                return ApproxPos

            x = np.linalg.inv(BtB) @ (np.transpose(ArrayB) @ ArrayL)

            # 更新坐标与钟差
            CurrentPos[0] += x[0]
            CurrentPos[1] += x[1]
            CurrentPos[2] += x[2]
            dtr += x[3] / c

            # ------ 6. 后验残差粗差剔除 --------------------------------
            # 计算各观测值残差 v = B*x - l
            residuals = ArrayB @ x - ArrayL
            gross_mask = np.abs(residuals) > GROSS_THR
            if np.any(gross_mask) and iteration < 10:
                # 将粗差卫星的伪距清零，下次迭代自动跳过
                for i, kid in enumerate(residual_idx):
                    if gross_mask[i]:
                        ObsPseudorange[kid] = 0
                # 不计入收敛，继续迭代
                continue

            # 收敛判断：坐标改正量 < 0.001m
            if np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < 0.001:
                break

        print(
            f"平差后的X坐标: {CurrentPos[0]:.3f}",
            f"平差后的Y坐标: {CurrentPos[1]:.3f}",
            f"平差后的Z坐标: {CurrentPos[2]:.3f}",
            f"(迭代次数: {iteration+1}, 有效卫星: {len(B)}颗)",
        )
        return CurrentPos
