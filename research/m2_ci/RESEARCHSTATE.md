# M2-CI 元数据索引

使用项目解释器从仓库根目录运行 `research/researchstate_m2_ci.py preview|refresh|verify`。

- `preview` 只读，校验历史链、原900项证据与新登记绑定，预览新阶段。
- `refresh` 将元数据追加至不可变 `research/state_history/`，再原子替换当前索引；相同内容不产生新revision。
- `verify` 校验当前索引、历史链、全部证据字节SHA及新登记派生一致性。

锚点为 revision 15，content SHA256：`6cfe6532d9078e68e91ae62e8a50cb2549d9bea1a38006bcbb2d813f77b55058`。原900项证据的顺序、角色、哈希、expected SHA与状态全部保留；原 experiments、supplementary_experiments 和 holdout 元数据不变，holdout 始终为 `CONSUMED`。

仅更新 current_stage、active_protocol、evidence、blockers、pending_actions，以及增加 `development_experiments.M2_CONDITIONAL_INCREMENT_01`。新登记必须绑定 config.yaml 精确SHA、TODO、附件副本与存在的原始附件来源SHA；同路径历史证据复用，不能删除或改写。登记状态 `IN_PROGRESS` / `in_progress` 映射为索引 `IN_PROGRESS`；`COMPLETE` / `development_complete` / `completed_development_research` 映射为索引 `COMPLETE`。

完成状态必须有 COMPLETE delivery_manifest.json、精确 experiment_id/protocol_sha256、CONSUMED holdout、非空 source_and_artifact_sha256，以及开发报告、未来正式协议草案、项目决策三个Markdown文件路径和SHA。登记与交付都必须声明 `future_formal_test_approved: false`。

此工具只读取JSON元数据并计算文件字节哈希；不读取数据数值、训练、评估、解析PDF或授予任何未来正式测试权限。旧 helper 的 derive/refresh 不适用于本阶段，不能用来覆盖M2索引。长期goal由会话工具维护；这里的 blockers 仅复制登记中的真实阻塞证据，不改变goal状态。
