# Replay PKL Viewer 使用说明

这个工具用于可视化 replay 目录下的 `replay_journal.pkl` 和 `episodes/*.pkl`。

## 1. 启动后端服务

在仓库根目录运行：

```bash
cd /home/lry/src/lry/lry-openpi-RLT
conda activate openpi_rlt
python rlt_online_rl/scripts/visualize_pkl/visualize_replay.py 
```

启动成功后会看到类似输出：

```text
Running on http://127.0.0.1:7860
```

## 2. 打开前端页面

在浏览器打开后端输出的地址：

```text
http://127.0.0.1:7860
```

前端页面由后端自动提供，不需要单独打开 `replay_viewer.html` 文件。

## 3. 端口被占用时

如果提示 `Address already in use`，换一个端口启动：

```bash
cd /home/lry/src/lry/lry-openpi-RLT
conda activate openpi_rlt
PORT=7862 python rlt_online_rl/scripts/visualize_pkl/visualize_replay.py
```

然后打开：

```text
http://127.0.0.1:7862
```

## 4. 切换 replay 目录

页面左上角有 `Replay 目录`：

- 可以手动输入 replay 目录的完整路径，然后点击 `打开`
- 也可以点击 `选择目录`，选择目录后再点击 `打开`

目标 replay 目录需要包含：

```text
replay_journal.pkl
episodes/*.pkl
```

## 5. 关闭服务

在启动服务的终端按：

```text
Ctrl+C
```
