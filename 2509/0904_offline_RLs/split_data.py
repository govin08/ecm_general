"""
Data preprocessing module for offline reinforcement learning
시계열 데이터의 train/test 분할과 강화학습용 데이터 준비 기능 제공
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def train_test_split_timeseries(data, test_ratio=0.2, time_column='Time'):
    """
    시계열 데이터를 시간 순서를 유지하며 train/test로 분할
    
    Parameters:
    -----------
    data : pd.DataFrame
        전체 시계열 데이터
    test_ratio : float, default=0.2
        테스트 데이터 비율 (0~1)
    time_column : str, default='Time'
        시간 컬럼명
        
    Returns:
    --------
    train_data : pd.DataFrame
        학습용 데이터 (시간 순서상 앞부분)
    test_data : pd.DataFrame
        테스트용 데이터 (시간 순서상 뒷부분)
    split_info : dict
        분할 정보 (인덱스, 비율 등)
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    
    # 시간 컬럼 처리
    if time_column in data.columns:
        # 시간 순서로 정렬
        if data[time_column].dtype == 'object':
            try:
                data = data.copy()
                data[time_column] = pd.to_datetime(data[time_column])
            except:
                print(f"Warning: Could not convert {time_column} to datetime, using original order")
        
        data = data.sort_values(time_column).reset_index(drop=True)
    else:
        print(f"Warning: {time_column} not found in data, using original order")
    
    # 분할 지점 계산
    total_samples = len(data)
    test_samples = int(total_samples * test_ratio)
    train_samples = total_samples - test_samples
    
    # 분할 실행
    train_data = data.iloc[:train_samples].copy()
    test_data = data.iloc[train_samples:].copy()
    
    # 분할 정보
    split_info = {
        'total_samples': total_samples,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'train_ratio': train_samples / total_samples,
        'test_ratio': test_samples / total_samples,
        'split_index': train_samples
    }
    
    print(f"Data split completed:")
    print(f"  Total samples: {total_samples}")
    print(f"  Train samples: {train_samples} ({split_info['train_ratio']:.1%})")
    print(f"  Test samples: {test_samples} ({split_info['test_ratio']:.1%})")
    
    if time_column in data.columns:
        print(f"  Train period: {train_data[time_column].iloc[0]} ~ {train_data[time_column].iloc[-1]}")
        print(f"  Test period: {test_data[time_column].iloc[0]} ~ {test_data[time_column].iloc[-1]}")
    
    return train_data, test_data, split_info

def prepare_rl_data(data, action_tag, target_tag, time_column='Time', reward_target_reduction=5.0):
    """
    시계열 데이터를 강화학습용 (s, a, r, s') 형태로 변환
    
    Parameters:
    -----------
    data : pd.DataFrame
        시계열 데이터
    action_tag : str
        행동 변수 컬럼명
    target_tag : str
        목표 변수 컬럼명 (보상 계산용)
    time_column : str, default='Time'
        시간 컬럼명
    reward_target_reduction : float, default=5.0
        목표 감소량 (ppm)
        
    Returns:
    --------
    rl_data : dict
        'states': np.array, 상태 데이터
        'actions': np.array, 행동 데이터  
        'rewards': np.array, 보상 데이터
        'next_states': np.array, 다음 상태 데이터
        'done': np.array, 종료 플래그
        'state_columns': list, 상태 변수 컬럼명들
        'target_idx': int, 목표 변수의 상태 내 인덱스
        'action_bounds': tuple, 행동 변수의 (min, max)
    """
    
    if action_tag not in data.columns:
        raise ValueError(f"Action column '{action_tag}' not found in data")
    if target_tag not in data.columns:
        raise ValueError(f"Target column '{target_tag}' not found in data")
    
    # 시간 컬럼 제외하고 feature 컬럼들만 선택
    feature_cols = [col for col in data.columns if col != time_column]
    df = data[feature_cols].copy()
    
    # 상태 변수들 (행동 변수 제외)
    state_columns = [col for col in feature_cols if col != action_tag]
    
    # 데이터 추출
    states = df[state_columns].values
    actions = df[[action_tag]].values
    
    # 다음 상태 생성 (한 타임스텝 이후)
    next_states = np.roll(states, -1, axis=0)
    
    # 마지막 샘플 제거 (다음 상태가 없으므로)
    states = states[:-1]
    actions = actions[:-1] 
    next_states = next_states[:-1]
    
    # 목표 변수의 인덱스 찾기
    try:
        target_idx = state_columns.index(target_tag)
    except ValueError:
        raise ValueError(f"Target column '{target_tag}' not found in state columns")
    
    # 보상 계산
    current_target = states[:, target_idx]
    next_target = next_states[:, target_idx]
    target_reduction = current_target - next_target
    
    # 목표 감소량에 가까울수록 높은 보상
    rewards = -np.abs(target_reduction - reward_target_reduction)
    
    # 추가 제약: 목표값이 너무 증가하면 패널티
    penalty_mask = next_target > current_target + 2.0
    rewards[penalty_mask] -= 10.0
    
    # 종료 플래그 (에피소드 구분이 없으므로 모두 False)
    done = np.zeros(len(states), dtype=bool)
    
    # 행동 변수 범위
    action_bounds = (float(np.min(actions)), float(np.max(actions)))
    
    rl_data = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'done': done,
        'state_columns': state_columns,
        'target_idx': target_idx,
        'action_bounds': action_bounds
    }
    
    print(f"RL data preparation completed:")
    print(f"  Samples: {len(states)}")
    print(f"  State dim: {states.shape[1]}")
    print(f"  Action dim: {actions.shape[1]}")
    print(f"  Target variable: {target_tag} (index: {target_idx})")
    print(f"  Action bounds: [{action_bounds[0]:.2f}, {action_bounds[1]:.2f}]")
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Reward std: {np.std(rewards):.3f}")
    
    return rl_data

def get_data_statistics(data, columns=None):
    """
    데이터의 기본 통계 정보 출력
    
    Parameters:
    -----------
    data : pd.DataFrame
        분석할 데이터
    columns : list, optional
        분석할 컬럼들 (None이면 모든 numeric 컬럼)
        
    Returns:
    --------
    stats : pd.DataFrame
        통계 정보
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = data[columns].describe()
    
    print("Data Statistics:")
    print("=" * 50)
    print(stats)
    print("=" * 50)
    
    return stats

def validate_rl_data(rl_data, verbose=True):
    """
    강화학습 데이터의 유효성 검사
    
    Parameters:
    -----------
    rl_data : dict
        prepare_rl_data에서 반환된 데이터
    verbose : bool, default=True
        상세 정보 출력 여부
        
    Returns:
    --------
    is_valid : bool
        데이터 유효성 여부
    validation_info : dict
        검증 결과 정보
    """
    validation_info = {}
    is_valid = True
    
    # 필수 키 존재 확인
    required_keys = ['states', 'actions', 'rewards', 'next_states', 'done', 'state_columns', 'target_idx', 'action_bounds']
    missing_keys = [key for key in required_keys if key not in rl_data]
    
    if missing_keys:
        validation_info['missing_keys'] = missing_keys
        is_valid = False
        if verbose:
            print(f"Error: Missing required keys: {missing_keys}")
    
    if is_valid:
        # 데이터 크기 일관성 확인
        n_samples = len(rl_data['states'])
        size_consistency = {
            'states': len(rl_data['states']) == n_samples,
            'actions': len(rl_data['actions']) == n_samples,
            'rewards': len(rl_data['rewards']) == n_samples,
            'next_states': len(rl_data['next_states']) == n_samples,
            'done': len(rl_data['done']) == n_samples
        }
        
        if not all(size_consistency.values()):
            validation_info['size_inconsistency'] = size_consistency
            is_valid = False
            if verbose:
                print(f"Error: Data size inconsistency: {size_consistency}")
        
        # NaN 값 확인
        nan_check = {
            'states': np.isnan(rl_data['states']).any(),
            'actions': np.isnan(rl_data['actions']).any(),
            'rewards': np.isnan(rl_data['rewards']).any(),
            'next_states': np.isnan(rl_data['next_states']).any()
        }
        
        if any(nan_check.values()):
            validation_info['nan_values'] = nan_check
            if verbose:
                print(f"Warning: NaN values found: {nan_check}")
        
        # 인덱스 범위 확인
        target_idx = rl_data['target_idx']
        state_dim = rl_data['states'].shape[1]
        
        if not (0 <= target_idx < state_dim):
            validation_info['invalid_target_idx'] = f"target_idx {target_idx} out of range [0, {state_dim-1}]"
            is_valid = False
            if verbose:
                print(f"Error: {validation_info['invalid_target_idx']}")
    
    validation_info['is_valid'] = is_valid
    
    if verbose and is_valid:
        print("✓ RL data validation passed successfully")
    
    return is_valid, validation_info

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    # 더미 데이터
    data_dict = {
        'Time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        '합성가스 유량': np.random.normal(100, 10, n_samples),
        '발전기 출력': np.random.normal(200, 20, n_samples),
        'DGAN Compressor 질소가스 유량': np.random.normal(150, 15, n_samples),
        '가스터빈 후단 질소산화물 농도': np.random.normal(50, 5, n_samples)
    }
    
    data = pd.DataFrame(data_dict)
    
    print("=== 데이터 전처리 모듈 테스트 ===\n")
    
    # 1. Train/Test 분할
    train_data, test_data, split_info = train_test_split_timeseries(data, test_ratio=0.2)
    
    print("\n")
    
    # 2. 강화학습 데이터 준비 (Train)
    action_tag = 'DGAN Compressor 질소가스 유량'
    target_tag = '가스터빈 후단 질소산화물 농도'
    
    train_rl_data = prepare_rl_data(train_data, action_tag, target_tag)
    
    print("\n")
    
    # 3. 데이터 검증
    is_valid, validation_info = validate_rl_data(train_rl_data)
    
    print(f"\n데이터 유효성: {'통과' if is_valid else '실패'}")