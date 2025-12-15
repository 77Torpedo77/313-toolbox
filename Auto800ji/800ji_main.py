from __future__ import annotations

import os
import sys
import json
from typing import Optional
import time
import random
import requests


CHAIR_NAME = "面向退化场景的多源信息融合自主建图与定位技术"
CHAIR_PERSONNEL_TOKEN = "xOIXrwAh5wqC3Btzz7YNpS3gg7u9gSUVB5sBCaZszVm2nj95kcaYMuTSrVqhTJveMs3yL4M1rXidltd9uOK9GXvF4h9XB7RZrU4bX8mIKg0oTDToEqgF0MsUqX2q8lNF"
max_retry_limit = 500

headers = {

	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 MicroMessenger/7.0.20.1781",
	"Accept": "application/json, text/plain, */*",
	"Content-Type": "application/x-www-form-urlencoded",
	"Origin": "https://app.nudt.edu.cn",
	"Referer": "https://app.nudt.edu.cn/chair/h5",
}



def fetch_chair_page(current: int = 1, timeout: int = 10) -> dict:
	"""获取单页的 chair 列表返回解析后的 JSON 字典。"""
	params = {"size": 10, "current": current, "lookCanApply": 1}
	cookies = {"chair-personnel-token": CHAIR_PERSONNEL_TOKEN}
	url = "https://app.nudt.edu.cn/chair/h5/api/chair/pageChair.do"
	with requests.Session() as s:
		resp = s.get(url, headers=headers, params=params, cookies=cookies, timeout=timeout)
		resp.raise_for_status()
		return resp.json()

def fetch_all_chairs() -> list:
	"""循环翻页获取所有 chairs，直到收集到 total 指定的数量。
	返回一个由所有 records 组成的列表。
	"""
	# 先获取第一页以获得 total
	page = 1
	all_records = []
	first = fetch_chair_page(current=page)
	# 解析结构，容错处理
	data = first.get("data") if isinstance(first, dict) else None
	if not data:
		return all_records
	total = data.get("total") or 0
	records = data.get("records") or []
	all_records.extend(records)


	page += 1
	while len(all_records) < total:
		time.sleep(1)
		p = fetch_chair_page(current=page)
		d = p.get("data") if isinstance(p, dict) else None
		if not d:
			break
		recs = d.get("records") or []
		if not recs:
			break
		all_records.extend(recs)
		page += 1

	# 截断到 total（以防服务返回多余数据）
	return all_records[:total]


def getId(chair_name: str, current: int) -> Optional[str]:
	"""根据 chairName 在 fetch_chair_page 返回的单页记录中查找 chairId。

	仅查询一次（current=1）。若找到完全匹配的 chairName，返回其 chairId；否则返回 None。
	"""
	p = fetch_chair_page(current=current)
	if not isinstance(p, dict):
		return None
	data = p.get("data")
	if not data:
		return None
	records = data.get("records") or []
	for r in records:
		name = r.get("chairName") or r.get("name")
		if name == chair_name:
			return r.get("chairId")

	return None
def send_chair_apply(chair_id: str, timeout: int = 10) -> dict:
	# 如果服务器依赖 Cookie 认证，优先使用 requests 的 cookies 参数传递
	cookies = {"chair-personnel-token": CHAIR_PERSONNEL_TOKEN}

	data = {"chairId": chair_id}

	with requests.Session() as s:
		resp = s.post("https://app.nudt.edu.cn/chair/h5/api/chair/chairApply.do", headers=headers, data=data, cookies=cookies, timeout=timeout)
		resp.raise_for_status()
		# 尝试解析 JSON 并返回
		return resp.json()
	
if __name__ == "__main__":

	for current in range(1,10):
		id = getId(CHAIR_NAME, current)
		if id is not None:
			print(f"《{CHAIR_NAME}》ID: {id}")
			break
		else:
			print(f"正在查找《{CHAIR_NAME}》，当前页码：{current}")
		time.sleep(random.randint(2, 4))

	for retry_num in range(max_retry_limit):
		print("======================================================================")
		print(f"正在申请  《{CHAIR_NAME}》...   当前次数：{retry_num}")
		res = send_chair_apply(chair_id=id)
		print(json.dumps(res, ensure_ascii=False, indent=2))

		code = res.get("code")
		if code == "00000":
			print(f"申请成功，代码：{code}")
			break

		
		time.sleep(random.uniform(3, 10))