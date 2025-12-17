"""
RAG问答机器人 - 主程序入口
提供命令行交互界面
"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.config_manager import config
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

def setup_logging():
    """配置日志（使用配置管理器）"""
    config.setup_logging()


def print_banner():
    """打印欢迎信息"""
    banner = """
    ╔═══════════════════════════════╗
    ║     RAG 智能问答机器人 v1.1      ║
    ║     基于检索增强生成技术          ║
    ╚═══════════════════════════════╝
    """
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
    可用命令：
    - 直接输入问题进行提问
    - help: 显示帮助信息
    - stats: 显示系统统计信息
    - clear: 清屏
    - exit/quit: 退出程序
    """
    print(help_text)


def print_stats(pipeline: RAGPipeline):
    """打印统计信息"""
    stats = pipeline.get_statistics()
    data_stats = stats.get('data_stats', {})

    print("\n" + "="*50)
    print("系统统计信息")
    print("="*50)

    if data_stats:
        print(f"📊 数据总量: {data_stats.get('total_count', 0)}")
        categories = data_stats.get('categories', {})
        if categories:
            print(f"📁 数据分类: {len(categories)} 个类别")
            # 显示前5个分类
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
            for cat, count in top_categories:
                print(f"   - {cat}: {count} 条")

    print(f"🔢 向量数量: {stats['vector_count']}")
    print(f"📐 向量维度: {stats['embedding_dim']}")
    print(f"🤖 向量化模型: {stats['model_name']}")
    print(f"💬 LLM模型: {stats['llm_model']}")
    print("="*50 + "\n")


def interactive_mode(pipeline: RAGPipeline):
    """交互式问答模式"""
    print("\n💡 提示: 输入 'help' 查看帮助，输入 'exit' 退出\n")

    while True:
        try:
            # 获取用户输入
            user_input = input("\n🙋 您的问题: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 感谢使用，再见！")
                break

            elif user_input.lower() == 'help':
                print_help()
                continue

            elif user_input.lower() == 'stats':
                print_stats(pipeline)
                continue

            elif user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue

            # 处理问题
            print("\n🔍 正在检索相关信息...")
            result = pipeline.answer(user_input, return_context=True)

            # 显示答案
            print(f"\n🤖 回答:\n{result['answer']}")

            # 显示元信息
            print(f"\n📊 检索到 {result['retrieved_count']} 个相关文档")
            print(f"📈 最高相似度: {result['top_similarity']:.2%}")

            # 可选：显示参考来源
            if result.get('context') and result['retrieved_count'] > 0:
                show_context = input("\n是否查看参考来源？(y/N): ").strip().lower()
                if show_context == 'y':
                    print("\n📚 参考来源:")
                    for i, doc in enumerate(result['context'][:3], 1):
                        print(f"\n[{i}] 相似度: {doc['similarity']:.2%}")
                        print(f"问题: {doc['qa_pair']['question']}")
                        print(f"答案: {doc['qa_pair']['answer'][:100]}...")

            # 添加明确的分隔符，让用户知道可以继续提问
            print("\n" + "―" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 检测到中断，正在退出...")
            break

        except Exception as e:
            logging.error(f"处理问题时出错: {e}", exc_info=True)
            print(f"\n❌ 抱歉，处理您的问题时出现错误: {str(e)}")
            print("💡 您可以继续输入其他问题\n")
            print("―" * 60)


def main():
    """主函数"""
    # 设置日志
    setup_logging()

    # 打印欢迎信息
    print_banner()

    try:
        # 初始化Pipeline（使用配置文件的默认值）
        print("⏳ 正在初始化系统...")
        pipeline = RAGPipeline()
        pipeline.initialize()
        print("✅ 系统初始化完成！\n")

        # 显示统计信息
        print_stats(pipeline)

        # 进入交互模式
        interactive_mode(pipeline)

    except Exception as e:
        logger.error(f"系统初始化失败: {e}", exc_info=True)
        print(f"\n❌ 系统初始化失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

