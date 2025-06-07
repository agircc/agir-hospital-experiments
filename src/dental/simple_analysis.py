#!/usr/bin/env python3
"""
Simple Dental Results Analysis
只显示每个模型的基本统计信息：正确率和答对数量
"""

import pandas as pd
from pathlib import Path

def analyze_dental_results():
    """分析牙科结果CSV文件的基本统计信息"""
    
    results_dir = Path('results/dental')
    
    # CSV文件列表
    csv_files = {
        'AGIR': 'agir_results.csv',
        'GPT-4.1-nano': 'gpt-4.1-nano_dental_results.csv', 
        'O3-mini': 'o3-mini_dental_results.csv'
    }
    
    print("牙科AI模型性能统计表")
    print("=" * 60)
    print(f"{'模型名称':<15} {'总题目数':<10} {'答对数量':<10} {'答错数量':<10} {'正确率':<10}")
    print("-" * 60)
    
    results = []
    
    for model_name, filename in csv_files.items():
        file_path = results_dir / filename
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            
            # 标准化 is_correct 列
            if 'is_correct' in df.columns:
                if df['is_correct'].dtype == 'object':
                    df['is_correct'] = df['is_correct'].map({
                        'True': True, 'False': False, 
                        1: True, 0: False, 
                        '1': True, '0': False
                    })
                else:
                    df['is_correct'] = df['is_correct'].astype(bool)
            
            total_questions = len(df)
            correct_answers = df['is_correct'].sum()
            incorrect_answers = total_questions - correct_answers
            accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0
            
            print(f"{model_name:<15} {total_questions:<10} {correct_answers:<10} {incorrect_answers:<10} {accuracy:<10.2f}%")
            
            results.append({
                'model': model_name,
                'total': total_questions,
                'correct': correct_answers,
                'incorrect': incorrect_answers,
                'accuracy': accuracy
            })
        else:
            print(f"{model_name:<15} 文件不存在")
    
    print("=" * 60)
    
    # 按正确率排序
    if results:
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        print(f"\n排名（按正确率）:")
        print("-" * 30)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['model']}: {result['accuracy']:.2f}%")

if __name__ == "__main__":
    analyze_dental_results() 