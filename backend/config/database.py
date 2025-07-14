from neo4j import GraphDatabase
import mysql.connector
from mysql.connector import pooling
import redis
from elasticsearch import Elasticsearch
from typing import Optional
import asyncio
from loguru import logger

from backend.config.settings import get_settings

# 全局数据库连接实例
neo4j_driver: Optional[GraphDatabase] = None
mysql_pool: Optional[pooling.MySQLConnectionPool] = None
redis_client: Optional[redis.Redis] = None
elasticsearch_client: Optional[Elasticsearch] = None

class Neo4jConnection:
    """Neo4j数据库连接管理"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
    
    def connect(self):
        """建立连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 测试连接
            self.driver.verify_connectivity()
            logger.info(f"Neo4j connected successfully to {self.uri}")
            return self.driver
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: dict = None):
        """执行查询"""
        if not self.driver:
            raise RuntimeError("Neo4j driver not connected")
        
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    
    def execute_write_transaction(self, query: str, parameters: dict = None):
        """执行写事务"""
        if not self.driver:
            raise RuntimeError("Neo4j driver not connected")
        
        with self.driver.session() as session:
            return session.write_transaction(lambda tx: tx.run(query, parameters))

class MySQLConnection:
    """MySQL数据库连接管理"""
    
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'use_unicode': True,
            'autocommit': True
        }
        self.pool = None
    
    def connect(self):
        """建立连接池"""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="equipment_fault_pool",
                pool_size=10,
                pool_reset_session=True,
                **self.config
            )
            logger.info("MySQL connection pool created successfully")
            return self.pool
        except Exception as e:
            logger.error(f"Failed to create MySQL connection pool: {e}")
            raise
    
    def get_connection(self):
        """获取连接"""
        if not self.pool:
            raise RuntimeError("MySQL connection pool not initialized")
        return self.pool.get_connection()
    
    def execute_query(self, query: str, params: tuple = None):
        """执行查询"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result
        finally:
            cursor.close()
            conn.close()
    
    def execute_update(self, query: str, params: tuple = None):
        """执行更新"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            cursor.close()
            conn.close()

class RedisConnection:
    """Redis连接管理"""
    
    def __init__(self, host: str, port: int, password: str, db: int):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.client = None
    
    def connect(self):
        """建立连接"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True
            )
            # 测试连接
            self.client.ping()
            logger.info(f"Redis connected successfully to {self.host}:{self.port}")
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")

class ElasticsearchConnection:
    """Elasticsearch连接管理"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None
    
    def connect(self):
        """建立连接"""
        try:
            self.client = Elasticsearch(
                [{'host': self.host, 'port': self.port}],
                timeout=30,
                retry_on_timeout=True
            )
            # 测试连接
            if self.client.ping():
                logger.info(f"Elasticsearch connected successfully to {self.host}:{self.port}")
                return self.client
            else:
                raise ConnectionError("Elasticsearch ping failed")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("Elasticsearch connection closed")

async def init_databases():
    """初始化所有数据库连接"""
    global neo4j_driver, mysql_pool, redis_client, elasticsearch_client
    
    settings = get_settings()
    
    try:
        # 初始化Neo4j连接
        neo4j_conn = Neo4jConnection(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD
        )
        neo4j_driver = neo4j_conn.connect()
        
        # 初始化MySQL连接池
        mysql_conn = MySQLConnection(
            settings.MYSQL_HOST,
            settings.MYSQL_PORT,
            settings.MYSQL_USER,
            settings.MYSQL_PASSWORD,
            settings.MYSQL_DATABASE
        )
        mysql_pool = mysql_conn.connect()
        
        # 初始化Redis连接
        redis_conn = RedisConnection(
            settings.REDIS_HOST,
            settings.REDIS_PORT,
            settings.REDIS_PASSWORD,
            settings.REDIS_DB
        )
        redis_client = redis_conn.connect()
        
        # 初始化Elasticsearch连接
        es_conn = ElasticsearchConnection(
            settings.ELASTICSEARCH_HOST,
            settings.ELASTICSEARCH_PORT
        )
        elasticsearch_client = es_conn.connect()
        
        logger.info("All database connections initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise

async def close_databases():
    """关闭所有数据库连接"""
    global neo4j_driver, mysql_pool, redis_client, elasticsearch_client
    
    if neo4j_driver:
        neo4j_driver.close()
        neo4j_driver = None
    
    if redis_client:
        redis_client.close()
        redis_client = None
    
    if elasticsearch_client:
        elasticsearch_client.close()
        elasticsearch_client = None
    
    logger.info("All database connections closed")

def get_neo4j_driver():
    """获取Neo4j驱动"""
    if not neo4j_driver:
        raise RuntimeError("Neo4j driver not initialized")
    return neo4j_driver

def get_mysql_pool():
    """获取MySQL连接池"""
    if not mysql_pool:
        raise RuntimeError("MySQL connection pool not initialized")
    return mysql_pool

def get_redis_client():
    """获取Redis客户端"""
    if not redis_client:
        raise RuntimeError("Redis client not initialized")
    return redis_client

def get_elasticsearch_client():
    """获取Elasticsearch客户端"""
    if not elasticsearch_client:
        raise RuntimeError("Elasticsearch client not initialized")
    return elasticsearch_client