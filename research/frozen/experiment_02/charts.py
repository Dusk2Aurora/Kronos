"""Actual UTC quarter facets and auditable CSV tables for locked RV experiment."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def save_figure(fig, out, name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def charts(run):
    run = Path(run)
    out = run/'figures'
    out.mkdir(exist_ok=True)
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    frame = pd.read_csv(run/'evaluated_predictions.csv')
    frame['decision_boundary_at'] = pd.to_datetime(frame.decision_boundary_at,utc=True)
    test = frame[frame.role == 'test'].copy()
    # Keep every registered family and validation-selected alias in all outputs.
    test['week'] = test.decision_boundary_at.dt.floor('D')-pd.to_timedelta(test.decision_boundary_at.dt.weekday,unit='D')
    weekly = test.groupby(['fold_id','family','week'],as_index=False).agg(actual_RV=('RV_effective','mean'),predicted_RV=('prediction','mean'),N=('prediction','size'))
    weekly.to_csv(out/'weekly_RV.csv',index=False)
    fig, axes = plt.subplots(4,1,figsize=(13,13),sharex=False)
    for ax, (fid, fold) in zip(axes,weekly.groupby('fold_id',sort=True)):
        actual = fold[fold.family=='R2']
        ax.plot(actual.week,actual.actual_RV,color='black',linewidth=2.2,label='实际4h RV')
        for family, model in fold.groupby('family',sort=True):
            ax.plot(model.week,model.predicted_RV,label=family,linewidth=1,alpha=.75)
        ax.set_yscale('log')
        ax.set_title(f'{fid} 测试季度：周均RV（每季度独立呈现）')
        ax.set_ylabel('4h平方对数收益 / 对数轴')
        ax.set_xlabel('真实UTC日期；标签有效RV；非年化')
        ax.grid(alpha=.2)
    axes[0].legend(ncol=5,fontsize=8)
    fig.suptitle('未来4h已实现方差与全部注册预测；来源：evaluated_predictions.csv')
    fig.tight_layout()
    save_figure(fig,out,'weekly_RV')
    daily = pd.read_csv(run/'paired_daily_gains.csv')
    daily['day'] = pd.to_datetime(daily.day,utc=True)
    daily = daily.sort_values(['fold_id','comparator','day'])
    daily['cumulative_QLIKE_gain'] = daily.groupby(['fold_id','comparator']).QLIKE_gain_sum.cumsum()
    daily.to_csv(out/'daily_cumulative_QLIKE_gains.csv',index=False)
    fig, axes = plt.subplots(4,1,figsize=(13,12))
    for ax, (fid,fold) in zip(axes,daily.groupby('fold_id',sort=True)):
        for comparator, series in fold.groupby('comparator',sort=True):
            ax.plot(series.day,series.cumulative_QLIKE_gain,label=comparator)
        ax.axhline(0,color='grey',linewidth=.7)
        ax.set_title(f'{fid} 日累计配对QLIKE增益：季度起点重置为0')
        ax.set_ylabel('累计损失差（控制−R2）')
        ax.set_xlabel('真实UTC日期；正值支持R2；不是收益/权益')
        ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=9)
    fig.suptitle('来源：paired_daily_gains.csv；随机控制为三种子逐机会损失平均')
    fig.tight_layout()
    save_figure(fig,out,'daily_cumulative_QLIKE_gains')
    metrics = pd.read_csv(run/'metrics.csv')
    metrics = metrics[metrics.role=='test'].copy()
    midpoint = test.groupby('fold_id').decision_boundary_at.agg(['min','max'])
    quarter_time = (midpoint['min'] + (midpoint['max']-midpoint['min'])/2).to_dict()
    metrics['quarter_midpoint_utc'] = metrics.fold_id.map(quarter_time)
    metrics.to_csv(out/'quarter_test_losses.csv',index=False)
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    for ax, loss in zip(axes,['QLIKE','Regret','logMSE']):
        for family, model in metrics.groupby('family',sort=True):
            ax.plot(model.quarter_midpoint_utc,model[loss],marker='o',label=family,alpha=.8)
        ax.set_title(f'各测试季度 {loss}')
        ax.set_xlabel('独立测试季度（非连续复利）')
        ax.set_ylabel('均值损失；越低越好')
        ax.grid(alpha=.2)
    axes[-1].legend(bbox_to_anchor=(1.01,1),fontsize=8)
    fig.suptitle('全部注册模型及validation选定控制；来源：metrics.csv')
    fig.tight_layout()
    save_figure(fig,out,'quarter_test_losses')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    print(charts(parser.parse_args().run))

if __name__=='__main__':
    main()
