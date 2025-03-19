import secrets
import string
import hashlib
import time
import base64
from datetime import datetime, timedelta
from supabase import create_client
import os
from typing import List, Dict, Optional

class KeyManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.VERSION = "1"
        self.KEY_LENGTH = 32
        self.supabase = create_client(supabase_url, supabase_key)
        
    def generate_random_string(self, length: int) -> str:
        """生成指定长度的加密安全随机字符串"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def create_timestamp(self) -> str:
        """创建时间戳编码"""
        return base64.b32encode(str(int(time.time())).encode()).decode()[:8]
    
    def generate_key(self, valid_days: int = 30) -> dict:
        """生成单个密钥及其信息"""
        random_part = self.generate_random_string(self.KEY_LENGTH)
        timestamp = self.create_timestamp()
        raw_key = f"{self.VERSION}-{timestamp}-{random_part}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        expiry_date = datetime.now() + timedelta(days=valid_days)
        
        return {
            "key_value": raw_key,
            "key_hash": key_hash,
            "expires_at": expiry_date.isoformat(),
            "version": self.VERSION,
            "is_active": True
        }
    
    def generate_and_store_keys(self, count: int = 20, valid_days: int = 30) -> List[Dict]:
        """生成多个密钥并存储到数据库"""
        keys = []
        for _ in range(count):
            key_info = self.generate_key(valid_days)
            # 将密钥信息插入数据库
            result = self.supabase.table('api_keys').insert(key_info).execute()
            keys.append(key_info)
        return keys
    
    def verify_key(self, key: str) -> Dict:
        """验证密钥是否有效"""
        try:
            # 计算输入密钥的哈希值
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            
            # 查询数据库
            result = self.supabase.table('api_keys')\
                .select('*')\
                .eq('key_hash', key_hash)\
                .eq('is_active', True)\
                .execute()
            
            if not result.data:
                return {"valid": False, "message": "密钥不存在或已失效"}
            
            key_info = result.data[0]
            
            # 检查是否过期
            expires_at = datetime.fromisoformat(key_info['expires_at'].replace('Z', '+00:00'))
            if expires_at < datetime.now(expires_at.tzinfo):
                # 更新密钥状态为失效
                self.supabase.table('api_keys')\
                    .update({"is_active": False})\
                    .eq('key_hash', key_hash)\
                    .execute()
                return {"valid": False, "message": "密钥已过期"}
            
            # 更新最后使用时间
            self.supabase.table('api_keys')\
                .update({"last_used_at": datetime.now().isoformat()})\
                .eq('key_hash', key_hash)\
                .execute()
            
            return {"valid": True, "message": "密钥有效"}
            
        except Exception as e:
            return {"valid": False, "message": f"验证过程出错: {str(e)}"}

def main():
    # 从环境变量获取 Supabase 配置
    supabase_url = ""
    supabase_key = ""
    
    if not supabase_url or not supabase_key:
        print("请设置 SUPABASE_URL 和 SUPABASE_KEY 环境变量")
        return
    
    # 创建密钥管理器实例
    manager = KeyManager(supabase_url, supabase_key)
    
    # 生成并存储20个密钥
    print("正在生成密钥...")
    keys = manager.generate_and_store_keys(count=20, valid_days=30)
    
    print(f"\n成功生成 {len(keys)} 个密钥:")
    for i, key in enumerate(keys, 1):
        print(f"\n密钥 {i}:")
        print(f"密钥值: {key['key_value']}")
        print(f"过期时间: {key['expires_at']}")
    
    # 测试验证功能
    test_key = keys[0]['key_value']
    print("\n测试验证第一个密钥:")
    result = manager.verify_key(test_key)
    print(f"验证结果: {result}")

if __name__ == "__main__":
    main()
