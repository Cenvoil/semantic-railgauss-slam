#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OSDaR23 序列 zip 下载器（走 anubis 挑战）。

用法: dl_osdar_seq.py <seq_name> <out_zip> [log]
例:   dl_osdar_seq.py 5_station_bergedorf_5.1 <PROJECT_ROOT>/data_osdar/5_station_bergedorf_5.1.zip

流程与 1.2 当初一致（见 anubis_dl.py）：
  1. GET zip URL -> 若返回 anubis 挑战页，解析 challenge JSON
  2. result = sha256_hex(randomData)
  3. GET pass-challenge（带 cookie）
  4. 用换回的 auth cookie 重新 GET zip，边下边写 + 定期回报进度

刻意不做的事：
  * 不做 Range 断点续传 —— 实测服务器忽略 Range（Content-Range 为 null），
    分段并发只会得到 8 个从头开始的完整响应，浪费带宽且无法拼接。
  * 不重试整包 —— 网络抖动交给 urllib 的重试；失败会打印已完成字节数，便于人工判断。
"""
import hashlib
import http.cookiejar
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://download.data.fid-move.de'
DL_DIR = '/dzsf/osdar23/'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

jar = http.cookiejar.CookieJar()
op = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(context=ctx),
    urllib.request.HTTPCookieProcessor(jar),
)
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

CHUNK = 8 << 20          # 8 MB 写盘粒度
REPORT = 30              # 每 30 s 回报一次


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def _req(url, headers=None):
    h = {'User-Agent': UA, 'Accept': '*/*'}
    if headers:
        h.update(headers)
    return op.open(urllib.request.Request(url, headers=h), timeout=120)


def solve_challenge(rel):
    """若首访被 anubis 拦，做一次挑战求解。返回是否真的拦过。"""
    r = _req(BASE + rel)
    ctype = r.headers.get('Content-Type', '')
    if 'zip' in ctype or 'octet' in ctype:
        return r, False
    html = r.read().decode('utf-8', 'replace')
    m = re.search(r'id="anubis_challenge"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise RuntimeError('既不是 zip 也不是 anubis 挑战页，ctype=%s\n%s' % (ctype, html[:400]))
    chal = json.loads(m.group(1))['challenge']
    log('收到 anubis 挑战 id=%s' % chal['id'])
    result = hashlib.sha256(chal['randomData'].encode('utf-8')).hexdigest()
    pc = ('/.within.website/x/cmd/anubis/api/pass-challenge?id=%s&redir=%s&result=%s'
          % (chal['id'], urllib.parse.quote(rel, safe=''), result))
    r3 = _req(BASE + pc)
    r3.read(300)
    log('挑战已提交，cookies=%s' % [c.name for c in jar])
    return None, True


def download(rel, out):
    r, blocked = solve_challenge(rel)
    if r is None:
        r = _req(BASE + rel)
    total = int(r.headers.get('Content-Length', 0))
    log('目标 %s  Content-Length=%s (%.2f GB)' % (rel, total, total / 2**30))
    got = 0
    t0 = time.time()
    last = t0
    with open(out, 'wb') as f:
        while True:
            buf = r.read(CHUNK)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            now = time.time()
            if now - last >= REPORT:
                rate = got / max(now - t0, 1e-6) / 2**20
                pct = 100.0 * got / total if total else 0.0
                eta = (total - got) / max(got / max(now - t0, 1e-6), 1e-6) if got else 0
                log('  %.2f GB / %.2f GB (%.1f%%)  %.1f MB/s  ETA %.1f min'
                    % (got / 2**30, total / 2**30, pct, rate, eta / 60))
                last = now
    log('下载结束 %d 字节，用时 %.1f min' % (got, (time.time() - t0) / 60))
    if total and got != total:
        raise RuntimeError('长度不符：期望 %d 实得 %d' % (total, got))
    with open(out, 'rb') as f:
        head = f.read(4)
    if head[:2] != b'PK':
        raise RuntimeError('不是 zip：头 %r' % head)
    return got


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    seq = sys.argv[1]
    out = sys.argv[2]
    logfile = sys.argv[3] if len(sys.argv) > 3 else None
    if os.path.exists(out) and os.path.getsize(out) > 0:
        log('目标已存在，跳过：%s (%d 字节)' % (out, os.path.getsize(out)))
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.part'
    rel = DL_DIR + seq + '.zip'
    try:
        download(rel, tmp)
        os.rename(tmp, out)      # 原子落位：完整才可见
        log('落位 %s' % out)
        log('DL_OK %s %d' % (seq, os.path.getsize(out)))
    except Exception as e:
        log('DL_FAIL %s %r  （已完成 %d 字节，临时文件 %s）'
            % (seq, e, os.path.getsize(tmp) if os.path.exists(tmp) else 0, tmp))
        raise


if __name__ == '__main__':
    main()
