"""
KGQA框架的Django视图
整合新的知识图谱问答框架到现有的Django应用中
"""

import json
import logging
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from kgqa_framework import FaultAnalyzer
from kgqa_framework.config import current_config
from kgqa_framework.models.entities import EquipmentInfo, UserQuery


# 全局故障分析器实例
analyzer = None
logger = logging.getLogger(__name__)


def init_analyzer():
    """初始化故障分析器"""
    global analyzer
    if analyzer is None:
        try:
            analyzer = FaultAnalyzer(
                neo4j_uri=current_config.NEO4J_URI,
                neo4j_username=current_config.NEO4J_USERNAME,
                neo4j_password=current_config.NEO4J_PASSWORD,
                case_database_path=current_config.CASE_DATABASE_PATH,
                vectorizer_path=current_config.VECTORIZER_PATH,
                stopwords_path=current_config.STOPWORDS_PATH,
                custom_dict_path=current_config.CUSTOM_DICT_PATH,
                enable_web_search=current_config.ENABLE_WEB_SEARCH
            )
            logger.info("KGQA故障分析器初始化成功")
        except Exception as e:
            logger.error(f"KGQA故障分析器初始化失败: {e}")
            analyzer = None
    return analyzer


def json_response(data, code=200, message="成功"):
    """标准JSON响应格式"""
    response_data = {
        "code": code,
        "msg": message,
        "data": data,
    }
    response = HttpResponse(
        json.dumps(response_data, ensure_ascii=False),
        content_type="application/json;charset=utf-8"
    )
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@csrf_exempt
@require_http_methods(["GET", "POST", "OPTIONS"])
def kgqa_diagnosis(request):
    """
    KGQA故障诊断接口
    
    参数:
        - question: 故障描述 (必需)
        - pinpai: 品牌 (可选)
        - xinghao: 型号 (可选)  
        - errorid: 故障代码 (可选)
        - relationList: 相关现象，用|分隔 (可选)
    """
    if request.method == "OPTIONS":
        return json_response({})
    
    try:
        # 获取参数
        if request.method == "POST":
            data = json.loads(request.body.decode('utf-8'))
            question = data.get('question', '')
            brand = data.get('pinpai', '')
            model = data.get('xinghao', '')
            error_code = data.get('errorid', '')
            relation_list = data.get('relationList', '')
        else:
            question = request.GET.get('question', '')
            brand = request.GET.get('pinpai', '')
            model = request.GET.get('xinghao', '')
            error_code = request.GET.get('errorid', '')
            relation_list = request.GET.get('relationList', '')
        
        # 验证必需参数
        if not question:
            return json_response(None, 400, "故障描述不能为空")
        
        # 初始化分析器
        analyzer = init_analyzer()
        if analyzer is None:
            return json_response(None, 500, "系统初始化失败")
        
        # 处理相关现象
        related_phenomena = []
        if relation_list:
            related_phenomena = [p.strip() for p in relation_list.split('|') if p.strip()]
        
        # 执行故障分析
        result = analyzer.analyze_fault(
            fault_description=question,
            brand=brand if brand else None,
            model=model if model else None,
            error_code=error_code if error_code else None,
            related_phenomena=related_phenomena
        )
        
        # 格式化返回结果
        response_data = {
            "confidence": round(result.confidence, 3),
            "causes": result.causes[:5],  # 限制返回前5个原因
            "solutions": result.solutions[:10],  # 限制返回前10个解决方案
            "reasoning_path": result.reasoning_path,
            "similar_cases": [
                {
                    "similarity": round(case.similarity, 3),
                    "description": case.description,
                    "solution": case.solution
                }
                for case in result.similar_cases[:3]  # 只返回前3个相似案例
            ],
            "recommendations": result.recommendations,
            "analysis_time": "实时分析完成"
        }
        
        return json_response(response_data)
        
    except Exception as e:
        logger.error(f"故障诊断处理失败: {e}")
        return json_response(None, 500, f"处理失败: {str(e)}")


@csrf_exempt
@require_http_methods(["GET", "POST", "OPTIONS"])
def kgqa_question_answer(request):
    """
    KGQA智能问答接口
    
    参数:
        - question: 问题 (必需)
    """
    if request.method == "OPTIONS":
        return json_response({})
    
    try:
        # 获取问题
        if request.method == "POST":
            data = json.loads(request.body.decode('utf-8'))
            question = data.get('question', '')
        else:
            question = request.GET.get('question', '')
        
        if not question:
            return json_response(None, 400, "问题不能为空")
        
        # 初始化分析器
        analyzer = init_analyzer()
        if analyzer is None:
            return json_response(None, 500, "系统初始化失败")
        
        # 将问题作为故障描述进行分析
        result = analyzer.analyze_fault(fault_description=question)
        
        # 格式化问答结果
        answer_data = {
            "question": question,
            "answer": result.solutions[0] if result.solutions else "未找到相关解决方案",
            "confidence": round(result.confidence, 3),
            "related_info": {
                "possible_causes": result.causes[:3],
                "additional_solutions": result.solutions[1:4] if len(result.solutions) > 1 else [],
                "recommendations": result.recommendations[:3]
            }
        }
        
        return json_response(answer_data)
        
    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        return json_response(None, 500, f"处理失败: {str(e)}")


@csrf_exempt
@require_http_methods(["POST", "OPTIONS"])
def kgqa_feedback(request):
    """
    用户反馈接口
    
    参数:
        - question: 原始故障描述
        - solution: 选择的解决方案
        - effectiveness: 有效性评分 (0-1)
        - pinpai: 品牌 (可选)
        - xinghao: 型号 (可选)
        - errorid: 故障代码 (可选)
    """
    if request.method == "OPTIONS":
        return json_response({})
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        question = data.get('question', '')
        solution = data.get('solution', '')
        effectiveness = data.get('effectiveness', 0.0)
        brand = data.get('pinpai', '')
        model = data.get('xinghao', '')
        error_code = data.get('errorid', '')
        
        if not question or not solution:
            return json_response(None, 400, "故障描述和解决方案不能为空")
        
        try:
            effectiveness = float(effectiveness)
            if not 0 <= effectiveness <= 1:
                return json_response(None, 400, "有效性评分必须在0-1之间")
        except ValueError:
            return json_response(None, 400, "有效性评分格式错误")
        
        # 初始化分析器
        analyzer = init_analyzer()
        if analyzer is None:
            return json_response(None, 500, "系统初始化失败")
        
        # 创建用户查询对象
        equipment_info = EquipmentInfo(
            brand=brand if brand else None,
            model=model if model else None,
            error_code=error_code if error_code else None
        )
        
        user_query = UserQuery(
            equipment_info=equipment_info,
            fault_description=question,
            related_phenomena=[],
            user_feedback=None
        )
        
        # 记录用户反馈
        analyzer.add_user_feedback(user_query, solution, effectiveness)
        
        return json_response({
            "feedback_recorded": True,
            "message": "用户反馈已记录，感谢您的反馈"
        })
        
    except Exception as e:
        logger.error(f"反馈处理失败: {e}")
        return json_response(None, 500, f"处理失败: {str(e)}")


@require_http_methods(["GET"])
def kgqa_status(request):
    """
    系统状态检查接口
    """
    try:
        analyzer = init_analyzer()
        if analyzer is None:
            return json_response({
                "system_status": "error",
                "message": "系统初始化失败"
            }, 500)
        
        status = analyzer.get_system_status()
        status["system_status"] = "running"
        status["framework_version"] = "1.0.0"
        
        return json_response(status)
        
    except Exception as e:
        logger.error(f"状态检查失败: {e}")
        return json_response(None, 500, f"状态检查失败: {str(e)}")


@csrf_exempt
@require_http_methods(["POST", "OPTIONS"])
def kgqa_autocomplete(request):
    """
    自动补全接口
    
    参数:
        - text: 输入文本
        - limit: 返回结果数量限制
    """
    if request.method == "OPTIONS":
        return json_response({})
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        text = data.get('text', '')
        limit = int(data.get('limit', 5))
        
        if not text:
            return json_response([], 200, "输入文本为空")
        
        # 初始化分析器
        analyzer = init_analyzer()
        if analyzer is None:
            return json_response(None, 500, "系统初始化失败")
        
        # 提取关键词作为补全建议
        keywords = analyzer.text_processor.extract_keywords(text, top_k=limit)
        
        suggestions = [
            {
                "text": keyword,
                "weight": round(weight, 3),
                "type": "keyword"
            }
            for keyword, weight in keywords
        ]
        
        # 添加一些常见的故障描述模板
        common_templates = [
            "设备启动时出现异常",
            "运行过程中发生故障",
            "报警信息显示",
            "温度异常升高",
            "振动超出正常范围"
        ]
        
        for template in common_templates:
            if text.lower() in template.lower():
                suggestions.append({
                    "text": template,
                    "weight": 0.5,
                    "type": "template"
                })
        
        # 限制返回数量
        suggestions = suggestions[:limit]
        
        return json_response(suggestions)
        
    except Exception as e:
        logger.error(f"自动补全处理失败: {e}")
        return json_response(None, 500, f"处理失败: {str(e)}")


def shutdown_analyzer():
    """关闭分析器（在应用关闭时调用）"""
    global analyzer
    if analyzer:
        try:
            analyzer.close()
            analyzer = None
            logger.info("KGQA故障分析器已关闭")
        except Exception as e:
            logger.error(f"关闭分析器失败: {e}")