"""
Author: Hongkun Luo
Description: PyQt5 图形界面 —— GNSS 伪距单点定位 (SPP)
"""

import sys
import os
import math
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QFileDialog,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QTextEdit, QMessageBox,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# ── 配置中文字体（使用系统内置的 Noto Sans CJK）──
def _setup_chinese_font():
    import matplotlib.font_manager as fm
    # 优先尝试 Noto Sans CJK，再回退到其他 CJK 字体
    cjk_names = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
                 "Droid Sans Fallback", "SimHei", "Microsoft YaHei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in cjk_names:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    # 万能回退：直接用文件路径
    for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
        if os.path.isfile(path):
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return

_setup_chinese_font()

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ──────────────────────────────────────────────
# 后台计算线程
# ──────────────────────────────────────────────
class SppWorker(QThread):
    """在独立线程中运行 SPP 解算，通过信号把结果发回 GUI"""

    # 信号: 每解算完一个历元就发一次
    epoch_done = pyqtSignal(dict)       # {'epoch_idx', 'time_str', 'x','y','z','iters','valid_sats'}
    progress   = pyqtSignal(int, int)   # (current, total)
    finished_ok = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, o_file, n_file):
        super().__init__()
        self.o_file = o_file
        self.n_file = n_file

    def run(self):
        try:
            self._solve()
            self.finished_ok.emit()
        except Exception as e:
            import traceback
            self.error_occurred.emit(traceback.format_exc())

    # ── 核心解算逻辑（从 position.py / readfile.py 提取，加进度回调）──
    def _solve(self):
        # 重置类变量，支持多次运行
        from readfile import ReadFile
        ReadFile.ApproxPos    = [None] * 3
        ReadFile.NLines       = None
        ReadFile.OLines       = None
        ReadFile.OHeaderLastLine = 0
        ReadFile.ObsTypes     = []

        from readfile import ReadFile
        from position import Position

        rf = ReadFile([self.o_file, self.n_file])
        rf.CaculateSatelites()

        pos = Position(
            rf.SateliteObservation,
            rf.PosName,
            rf.Time,
            rf.SateliteClockCorrect,
        )

        # ── 逐历元解算（内联 MatchObservationAndCaculate，加进度信号）──
        lines      = pos.Lines
        lps        = pos.LinesPerSat
        NReadLine  = pos.OHeaderLastLine
        line       = lines[NReadLine]
        CurrentPos = list(pos.ApproxPos)

        # 预计总历元数（粗估：每历元至少占 lps*4+1 行）
        total_lines = len(lines)
        epoch_idx   = 0

        while line != "":
            if line.strip() == "":
                NReadLine += 1
                if NReadLine >= len(lines): break
                line = lines[NReadLine]; continue

            if line[0] != ' ' or not line[1:3].strip().isdigit():
                NReadLine += 1
                if NReadLine >= len(lines): break
                line = lines[NReadLine]; continue

            # 解析时间
            try:
                ot = [None]*6
                ot[0] = int(line[1:3].strip())  + 2000
                ot[1] = int(line[4:6].strip())
                ot[2] = int(line[7:9].strip())
                ot[3] = int(line[10:12].strip())
                ot[4] = int(line[13:15].strip())
                ot[5] = float(line[15:26].strip())
            except ValueError:
                NReadLine += 1
                if NReadLine >= len(lines): break
                line = lines[NReadLine]; continue

            num_sat = int(line[30:32])
            n_hdr   = int(num_sat / 12) + (0 if num_sat % 12 == 0 else 1)

            prn_str = ""
            for j in range(n_hdr):
                prn_str += lines[NReadLine + j][32:68].strip()
            obs_prn = [prn_str[k*3:k*3+3] for k in range(num_sat)]

            NReadLine += n_hdr

            # 读伪距
            pseudo = []
            for j in range(NReadLine, NReadLine + lps * num_sat, lps):
                p = 0
                if pos.P1ColRow >= 0:
                    pl = lines[j + pos.P1ColRow]
                    s  = pl[pos.P1ColOffset:pos.P1ColOffset+14].strip() if len(pl) > pos.P1ColOffset+14 else ""
                    if s: p = float(s)
                if p == 0 and pos.C1ColRow >= 0:
                    cl = lines[j + pos.C1ColRow]
                    s  = cl[pos.C1ColOffset:pos.C1ColOffset+14].strip() if len(cl) > pos.C1ColOffset+14 else ""
                    if s: p = float(s)
                pseudo.append(p)

            sat_xyz = pos.MatchToSatlite(ot, obs_prn)

            result, iters, valid = self._least_squares(pseudo, sat_xyz, CurrentPos)

            NReadLine += lps * num_sat

            if result is not None:
                CurrentPos = result
                time_str = "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:06.3f}".format(*ot)
                self.epoch_done.emit({
                    "epoch_idx":  epoch_idx,
                    "time_str":   time_str,
                    "x": result[0], "y": result[1], "z": result[2],
                    "iters":      iters,
                    "valid_sats": valid,
                })
                epoch_idx += 1

            self.progress.emit(NReadLine, total_lines)

            if NReadLine >= len(lines): break
            line = lines[NReadLine]

    # ── 迭代最小二乘（返回 (pos_list, iters, valid_count) 或 None）──
    def _least_squares(self, pseudo, sat_xyz, approx):
        c   = 299792458  # Bug修复12: 原值 3.0e8 近似误差约0.07%，改用精确光速值
        N   = len(sat_xyz)
        valid = sum(1 for k in range(N) if sat_xyz[k][3]==1 and pseudo[k]!=0)
        if valid < 4:
            return None, 0, valid

        cur  = list(approx)
        iters = 0
        for iters in range(10):
            B, L = [], []
            for k in range(N):
                if sat_xyz[k][3] == 0 or pseudo[k] == 0:
                    continue
                dx = sat_xyz[k][0]-cur[0]
                dy = sat_xyz[k][1]-cur[1]
                dz = sat_xyz[k][2]-cur[2]
                P0 = math.sqrt(dx*dx + dy*dy + dz*dz)
                B.append([-dx/P0, -dy/P0, -dz/P0, 1.0])
                L.append(pseudo[k] - P0 + c*sat_xyz[k][4])

            AB = np.array(B); AL = np.array(L)
            BtB = AB.T @ AB
            if np.linalg.matrix_rank(BtB) < 4:
                return None, iters+1, valid
            x = np.linalg.inv(BtB) @ (AB.T @ AL)
            cur[0] += x[0]; cur[1] += x[1]; cur[2] += x[2]
            if math.sqrt(x[0]**2+x[1]**2+x[2]**2) < 0.001:
                break
        return cur, iters+1, valid


# ──────────────────────────────────────────────
# 主窗口
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNSS 伪距单点定位 (SPP)")
        self.resize(1300, 820)
        self.results = []      # [{time_str, x, y, z, iters, valid_sats}, ...]
        self.worker  = None
        self._build_ui()

    # ──────────── 布局 ────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 左侧控制面板
        left = self._build_left_panel()
        root.addWidget(left, 0)

        # 右侧标签页（结果展示）
        self.tabs = self._build_tabs()
        root.addWidget(self.tabs, 1)

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

    # ── 左侧控制面板 ──
    def _build_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(10)

        # ── 文件选择组 ──
        grp_file = QGroupBox("输入文件")
        g_lay = QGridLayout(grp_file)

        g_lay.addWidget(QLabel("O 文件:"), 0, 0)
        self.le_ofile = QLineEdit()
        self.le_ofile.setPlaceholderText("观测文件 .23o / .obs")
        g_lay.addWidget(self.le_ofile, 1, 0)
        btn_o = QPushButton("浏览…")
        btn_o.clicked.connect(self._browse_ofile)
        g_lay.addWidget(btn_o, 1, 1)

        g_lay.addWidget(QLabel("N 文件:"), 2, 0)
        self.le_nfile = QLineEdit()
        self.le_nfile.setPlaceholderText("导航文件 .23n / .nav")
        g_lay.addWidget(self.le_nfile, 3, 0)
        btn_n = QPushButton("浏览…")
        btn_n.clicked.connect(self._browse_nfile)
        g_lay.addWidget(btn_n, 3, 1)

        # 快速填入示例文件
        btn_demo = QPushButton("载入示例数据")
        btn_demo.clicked.connect(self._load_demo)
        g_lay.addWidget(btn_demo, 4, 0, 1, 2)

        layout.addWidget(grp_file)

        # ── 运行控制组 ──
        grp_run = QGroupBox("解算控制")
        r_lay = QVBoxLayout(grp_run)

        self.btn_run = QPushButton("▶  开始解算")
        self.btn_run.setFixedHeight(38)
        self.btn_run.setFont(QFont("", 11, QFont.Bold))
        self.btn_run.setStyleSheet(
            "QPushButton{background:#2196F3;color:white;border-radius:4px;}"
            "QPushButton:hover{background:#1976D2;}"
            "QPushButton:disabled{background:#B0BEC5;}"
        )
        self.btn_run.clicked.connect(self._run)
        r_lay.addWidget(self.btn_run)

        self.btn_stop = QPushButton("■  停止")
        self.btn_stop.setFixedHeight(30)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#F44336;color:white;border-radius:4px;}"
            "QPushButton:hover{background:#D32F2F;}"
            "QPushButton:disabled{background:#B0BEC5;}"
        )
        self.btn_stop.clicked.connect(self._stop)
        r_lay.addWidget(self.btn_stop)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        r_lay.addWidget(self.progress_bar)

        layout.addWidget(grp_run)

        # ── 统计信息组 ──
        grp_stat = QGroupBox("解算统计")
        s_lay = QGridLayout(grp_stat)
        s_lay.setSpacing(4)

        def stat_row(label, row):
            lbl = QLabel(label)
            val = QLabel("—")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setFont(QFont("Courier", 9))
            s_lay.addWidget(lbl, row, 0)
            s_lay.addWidget(val, row, 1)
            return val

        self.lbl_epochs   = stat_row("历元总数:",    0)
        self.lbl_outliers = stat_row("粗差剔除:",    1)
        self.lbl_mean_x   = stat_row("X 均值:",      2)
        self.lbl_mean_y   = stat_row("Y 均值:",      3)
        self.lbl_mean_z   = stat_row("Z 均值:",      4)
        self.lbl_std_x    = stat_row("X 中误差:",    5)
        self.lbl_std_y    = stat_row("Y 中误差:",    6)
        self.lbl_std_z    = stat_row("Z 中误差:",    7)
        self.lbl_mean3d   = stat_row("3D 中误差:",   8)

        layout.addWidget(grp_stat)

        # ── 日志 ──
        grp_log = QGroupBox("日志")
        l_lay = QVBoxLayout(grp_log)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Courier", 8))
        self.log_box.setFixedHeight(160)
        l_lay.addWidget(self.log_box)
        layout.addWidget(grp_log)

        layout.addStretch()
        return panel

    # ── 右侧标签页 ──
    def _build_tabs(self):
        tabs = QTabWidget()

        # Tab 0: 结果表格
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["#", "时间", "X (m)", "Y (m)", "Z (m)", "迭代次数", "有效卫星"]
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for c in [2,3,4]:
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs.addTab(self.table, "📋 结果表格")

        # Tab 1: XYZ 时序图
        tabs.addTab(self._build_xyz_tab(), "📈 XYZ 时序")

        # Tab 2: 平面散点图（水平偏差）
        tabs.addTab(self._build_scatter_tab(), "🗺 平面散点")

        # Tab 3: 3D误差时序
        tabs.addTab(self._build_error_tab(), "📉 误差时序")

        return tabs

    def _make_fig_widget(self, nrows=1, ncols=1, figsize=(10,4.5)):
        fig  = Figure(figsize=figsize, tight_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, None)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2,2,2,2)
        v.addWidget(toolbar)
        v.addWidget(canvas)
        axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        return w, fig, canvas, axes

    def _build_xyz_tab(self):
        self._xyz_widget, self._xyz_fig, self._xyz_canvas, axes = \
            self._make_fig_widget(3, 1, (10, 6))
        self._ax_x, self._ax_y, self._ax_z = axes
        for ax, lbl in zip(axes, ["X (m)", "Y (m)", "Z (m)"]):
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        self._ax_z.set_xlabel("历元序号")
        return self._xyz_widget

    def _build_scatter_tab(self):
        self._scat_widget, self._scat_fig, self._scat_canvas, axes = \
            self._make_fig_widget(1, 1, (8, 7))
        self._ax_scat = axes[0]
        self._ax_scat.set_xlabel("ΔX (m)  东西方向")
        self._ax_scat.set_ylabel("ΔY (m)  南北方向")
        self._ax_scat.set_aspect("equal", "datalim")
        self._ax_scat.grid(True, alpha=0.3)
        self._ax_scat.set_title("水平分量散点图（相对于坐标均值）")
        return self._scat_widget

    def _build_error_tab(self):
        self._err_widget, self._err_fig, self._err_canvas, axes = \
            self._make_fig_widget(1, 1, (10, 4.5))
        self._ax_err = axes[0]
        self._ax_err.set_xlabel("历元序号")
        self._ax_err.set_ylabel("3D 误差 (m)  (相对均值)")
        self._ax_err.set_title("各历元 3D 定位误差（以所有历元均值为参考）")
        self._ax_err.grid(True, alpha=0.3)
        return self._err_widget

    # ──────────── 交互槽 ────────────
    def _browse_ofile(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 O 文件", "./data",
            "RINEX Obs (*.??o *.obs *.23o);;All (*)"
        )
        if p: self.le_ofile.setText(p)

    def _browse_nfile(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 N 文件", "./data",
            "RINEX Nav (*.??n *.nav *.23n);;All (*)"
        )
        if p: self.le_nfile.setText(p)

    def _load_demo(self):
        base = os.path.dirname(os.path.abspath(__file__))
        self.le_ofile.setText(os.path.join(base, "data", "al2h3340.23o"))
        self.le_nfile.setText(os.path.join(base, "data", "al2h3340.23n"))
        self._log("已载入示例数据（AL2H 测站）")

    def _log(self, msg):
        self.log_box.append(msg)

    def _run(self):
        o = self.le_ofile.text().strip()
        n = self.le_nfile.text().strip()
        if not o or not os.path.isfile(o):
            QMessageBox.warning(self, "错误", "请先选择有效的 O 文件")
            return
        if not n or not os.path.isfile(n):
            QMessageBox.warning(self, "错误", "请先选择有效的 N 文件")
            return

        # 清空上次结果
        self.results.clear()
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self._clear_plots()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.showMessage("解算中…")
        self._log(f"开始解算: {os.path.basename(o)}  +  {os.path.basename(n)}")

        self.worker = SppWorker(o, n)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self._log("已手动停止")
            self._on_finished()

    # ──────────── 信号槽（后台回调）────────────
    def _on_epoch(self, d):
        self.results.append(d)
        row = self.table.rowCount()
        self.table.insertRow(row)

        def item(txt, align=Qt.AlignRight|Qt.AlignVCenter):
            it = QTableWidgetItem(str(txt))
            it.setTextAlignment(align)
            return it

        self.table.setItem(row, 0, item(str(d["epoch_idx"]+1)))
        self.table.setItem(row, 1, item(d["time_str"], Qt.AlignLeft|Qt.AlignVCenter))
        self.table.setItem(row, 2, item(f"{d['x']:.3f}"))
        self.table.setItem(row, 3, item(f"{d['y']:.3f}"))
        self.table.setItem(row, 4, item(f"{d['z']:.3f}"))
        self.table.setItem(row, 5, item(str(d["iters"])))
        self.table.setItem(row, 6, item(str(d["valid_sats"])))
        # 每 50 个历元滚动一次，减少 GUI 刷新开销
        if row % 50 == 0:
            self.table.scrollToBottom()

    def _on_progress(self, cur, total):
        if total > 0:
            self.progress_bar.setValue(int(cur / total * 100))

    def _on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        n = len(self.results)
        self.status.showMessage(f"解算完成，共 {n} 个历元")
        self._log(f"解算完成，共 {n} 个历元")
        self._update_plots()
        self._update_stats()

    def _on_error(self, tb):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.showMessage("解算出错！")
        self._log("错误:\n" + tb)
        QMessageBox.critical(self, "解算错误", tb[:600])

    # ──────────── 图表刷新 ────────────
    def _clear_plots(self):
        for ax in [self._ax_x, self._ax_y, self._ax_z,
                   self._ax_scat, self._ax_err]:
            ax.cla()
        for ax in [self._ax_x, self._ax_y, self._ax_z]:
            ax.grid(True, alpha=0.3)
        self._ax_err.grid(True, alpha=0.3)
        self._ax_scat.grid(True, alpha=0.3)
        self._ax_z.set_xlabel("历元序号")
        self._ax_err.set_xlabel("历元序号")
        self._ax_err.set_ylabel("3D 误差 (m)")
        self._ax_scat.set_xlabel("ΔX (m)")
        self._ax_scat.set_ylabel("ΔY (m)")
        self._ax_scat.set_aspect("equal", "datalim")
        for c, ax in zip(["X","Y","Z"], [self._ax_x, self._ax_y, self._ax_z]):
            ax.set_ylabel(f"{c} (m)")
        self._xyz_canvas.draw()
        self._scat_canvas.draw()
        self._err_canvas.draw()

    def _update_plots(self):
        if not self.results:
            return
        xs_all = np.array([r["x"] for r in self.results])
        ys_all = np.array([r["y"] for r in self.results])
        zs_all = np.array([r["z"] for r in self.results])

        # ── 粗差剔除：以全部历元的中值为参考，|ΔX|>10m 或 |ΔY|>10m 则剔除 ──
        THRESHOLD = 10.0          # 单位：米
        ref_x = np.median(xs_all)
        ref_y = np.median(ys_all)
        mask_good = (np.abs(xs_all - ref_x) <= THRESHOLD) & \
                    (np.abs(ys_all - ref_y) <= THRESHOLD)
        mask_bad  = ~mask_good

        xs = xs_all[mask_good]
        ys = ys_all[mask_good]
        zs = zs_all[mask_good]
        idx_good = np.where(mask_good)[0]   # 保留历元在原始序号中的位置
        idx_bad  = np.where(mask_bad)[0]

        n_bad  = int(mask_bad.sum())
        n_good = int(mask_good.sum())

        # 把剔除计数写回统计面板（此处提前写，_update_stats 再写一次也无妨）
        self.lbl_outliers.setText(
            f"{n_bad} 个  ({n_bad/len(xs_all)*100:.1f}%)"
        )
        self.lbl_outliers.setStyleSheet(
            "color: #D32F2F; font-weight:bold;" if n_bad > 0 else ""
        )
        if n_good == 0:
            self._log("警告：剔除粗差后无有效历元，请检查数据。")
            return

        mx, my, mz = xs.mean(), ys.mean(), zs.mean()

        # ── XYZ 时序（好点蓝/绿/橙，粗差红叉）──
        for ax, arr_all, arr, label, color in zip(
            [self._ax_x, self._ax_y, self._ax_z],
            [xs_all, ys_all, zs_all],
            [xs, ys, zs],
            ["X (m)", "Y (m)", "Z (m)"],
            ["#1976D2", "#388E3C", "#F57C00"],
        ):
            ax.cla()
            ax.plot(idx_good, arr, color=color, linewidth=0.8, alpha=0.9,
                    label=f"有效 ({n_good})")
            ax.axhline(arr.mean(), color="red", linewidth=1, linestyle="--",
                       label=f"均值 {arr.mean():.3f}")
            if n_bad > 0:
                ax.scatter(idx_bad, arr_all[mask_bad], color="#D32F2F",
                           marker="x", s=30, linewidths=1.2, zorder=5,
                           label=f"粗差 ({n_bad})")
            ax.set_ylabel(label)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
        self._ax_z.set_xlabel("历元序号")
        self._xyz_canvas.draw()

        # ── 水平散点（仅好点，粗差用红叉单独标出）──
        self._ax_scat.cla()
        dx = xs - mx
        dy = ys - my
        sc = self._ax_scat.scatter(dx, dy, c=idx_good, cmap="plasma",
                                   s=4, alpha=0.7, label=f"有效 ({n_good})")
        # 粗差点相对于好点均值的偏差（可能很远，仅在图内可见范围标注）
        if n_bad > 0:
            dx_bad = xs_all[mask_bad] - mx
            dy_bad = ys_all[mask_bad] - my
            self._ax_scat.scatter(dx_bad, dy_bad, color="#D32F2F",
                                  marker="x", s=60, linewidths=1.5, zorder=5,
                                  label=f"粗差 ({n_bad})")
        self._ax_scat.axhline(0, color="gray", linewidth=0.8)
        self._ax_scat.axvline(0, color="gray", linewidth=0.8)
        # 绘制10m剔除阈值框（以好点均值为中心）
        from matplotlib.patches import Rectangle
        rect = Rectangle((-THRESHOLD, -THRESHOLD), 2*THRESHOLD, 2*THRESHOLD,
                         linewidth=1.2, edgecolor="#FF6F00", facecolor="none",
                         linestyle="--", label=f"±{THRESHOLD:.0f} m 阈值")
        self._ax_scat.add_patch(rect)
        self._ax_scat.set_xlabel("ΔX (m)  东西方向")
        self._ax_scat.set_ylabel("ΔY (m)  南北方向")
        self._ax_scat.set_title(
            f"水平分量散点图（相对坐标均值，±{THRESHOLD:.0f} m 阈值剔除粗差）"
        )
        self._ax_scat.set_aspect("equal", "datalim")
        self._ax_scat.grid(True, alpha=0.3)
        self._ax_scat.legend(fontsize=7, loc="upper right")
        self._scat_fig.colorbar(sc, ax=self._ax_scat, label="历元序号")
        self._scat_canvas.draw()

        # ── 3D 误差时序（以好点均值为参考，仅绘制好点）──
        err3d = np.sqrt((xs-mx)**2 + (ys-my)**2 + (zs-mz)**2)
        # 3D 中误差 = sqrt(σX²+σY²+σZ²)
        rmse_x = np.sqrt(np.mean((xs-mx)**2))
        rmse_y = np.sqrt(np.mean((ys-my)**2))
        rmse_z = np.sqrt(np.mean((zs-mz)**2))
        rmse_3d = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)
        self._ax_err.cla()
        self._ax_err.plot(idx_good, err3d, color="#7B1FA2", linewidth=0.8, alpha=0.9,
                          label="各历元偏差")
        self._ax_err.axhline(rmse_3d, color="red", linewidth=1.2,
                             linestyle="--", label=f"3D中误差 {rmse_3d:.3f} m")
        self._ax_err.set_xlabel("历元序号")
        self._ax_err.set_ylabel("3D 偏差 (m)")
        self._ax_err.set_title(
            f"各历元 3D 定位偏差（剔除粗差后，共 {n_good} 个历元，3D中误差 {rmse_3d:.3f} m）"
        )
        self._ax_err.legend(fontsize=9)
        self._ax_err.grid(True, alpha=0.3)
        self._err_canvas.draw()

        self._log(
            f"粗差剔除完成：共 {len(xs_all)} 历元，"
            f"剔除 {n_bad} 个（|ΔX|>{THRESHOLD:.0f}m 或 |ΔY|>{THRESHOLD:.0f}m），"
            f"保留 {n_good} 个。"
        )

    def _update_stats(self):
        if not self.results:
            return
        xs_all = np.array([r["x"] for r in self.results])
        ys_all = np.array([r["y"] for r in self.results])
        zs_all = np.array([r["z"] for r in self.results])

        # 与 _update_plots 保持一致的剔除逻辑
        THRESHOLD = 10.0
        ref_x = np.median(xs_all)
        ref_y = np.median(ys_all)
        mask_good = (np.abs(xs_all - ref_x) <= THRESHOLD) & \
                    (np.abs(ys_all - ref_y) <= THRESHOLD)
        xs = xs_all[mask_good]
        ys = ys_all[mask_good]
        zs = zs_all[mask_good]
        n_bad = int((~mask_good).sum())

        if len(xs) == 0:
            return

        mx, my, mz = xs.mean(), ys.mean(), zs.mean()
        # 各分量中误差 = sqrt(Σ(Xi−X̄)²/n)，即 ddof=0 的标准差
        rmse_x = np.sqrt(np.mean((xs - mx)**2))
        rmse_y = np.sqrt(np.mean((ys - my)**2))
        rmse_z = np.sqrt(np.mean((zs - mz)**2))
        # 3D 中误差 = sqrt(σX² + σY² + σZ²)
        rmse_3d = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)

        self.lbl_epochs.setText(f"{len(xs_all)}  (有效 {len(xs)})")
        self.lbl_outliers.setText(
            f"{n_bad} 个  ({n_bad/len(xs_all)*100:.1f}%)"
        )
        self.lbl_outliers.setStyleSheet(
            "color: #D32F2F; font-weight:bold;" if n_bad > 0 else ""
        )
        self.lbl_mean_x.setText(f"{mx:.3f} m")
        self.lbl_mean_y.setText(f"{my:.3f} m")
        self.lbl_mean_z.setText(f"{mz:.3f} m")
        self.lbl_std_x.setText(f"{rmse_x:.3f} m")
        self.lbl_std_y.setText(f"{rmse_y:.3f} m")
        self.lbl_std_z.setText(f"{rmse_z:.3f} m")
        self.lbl_mean3d.setText(f"{rmse_3d:.3f} m")


# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
