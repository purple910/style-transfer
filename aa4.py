"""
    @Time    : 2021/2/19 18:47 
    @Author  : fate
    @Site    : 
    @File    : aa4.py
    @Software: PyCharm
"""
import os

'''
var_list：Variable/ 的列表SaveableObject，或者将名称映射到SaveableObjects 的字典。如果None，默认为所有可保存对象的列表。
reshape：If True，允许从变量具有不同形状的检查点恢复参数。
sharded：如果True，将每个设备分成一个检查点。
max_to_keep：要保留的最近检查点的最大数量。默认为5。
keep_checkpoint_every_n_hours：保持检查站的频率。默认为10,000小时。
name：字符串。添加操作时用作前缀的可选名称。
restore_sequentially：A Bool，如果为true，则导致不同变量的恢复在每个设备中顺序发生。这可以在恢复非常大的模型时降低内存使用量。
saver_def：SaverDef使用可选的proto而不是运行构建器。这仅适用于想要Saver为先前构建的Graph具有a 的对象重新创建对象的专业代码Saver。该saver_def原型应该是返回一个 as_saver_def()的电话Saver说是为创建Graph。
builder：SaverBuilder如果saver_def未提供，则可以选择使用。默认为BulkSaverBuilder()。
defer_build：如果True，请将保存和恢复操作添加到 build()呼叫中。在这种情况下，build()应在最终确定图表或使用保护程序之前调用。
allow_empty：如果False（默认）如果图中没有变量则引发错误。否则，无论如何构建保护程序并使其成为无操作。
write_version：控制保存检查点时使用的格式。它还会影响某些文件路径匹配逻辑。V2格式是推荐的选择：它在恢复期间所需的内存和延迟方面比V1更加优化。无论此标志如何，Saver都能够从V2和V1检查点恢复。
pad_step_number：如果为True，则将检查点文件路径中的全局步骤编号填充到某个固定宽度（默认为8）。默认情况下这是关闭的。
save_relative_paths：如果True，将写入检查点状态文件的相对路径。如果用户想要复制检查点目录并从复制的目录重新加载，则需要这样做。
filename：如果在图形构造时知道，则用于变量加载/保存的文件名。
'''
import tensorflow as tf

