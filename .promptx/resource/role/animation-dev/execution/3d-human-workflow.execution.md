<execution>
  <constraint>
    ## 技术硬约束
    - **内存限制**：单个SMPL-X模型约50MB，批处理需考虑GPU内存
    - **精度要求**：关节旋转精度至少到0.01弧度
    - **实时约束**：交互预览需保持30FPS以上
    - **格式兼容**：必须支持FBX/GLTF/USD导出
    - **Python版本**：Blender内置Python与系统Python版本差异
  </constraint>
  
  <rule>
    ## 强制执行规则
    - **坐标系转换**：所有导入必须进行Y-up到Z-up转换
    - **单位统一**：内部计算统一使用米，输出时转换
    - **权重归一化**：蒙皮权重和必须为1.0
    - **拓扑保持**：变形不改变网格拓扑结构
    - **版本锁定**：指定SMPL-X模型版本避免不兼容
  </rule>
  
  <guideline>
    ## 最佳实践指南
    - **增量更新**：只更新变化的顶点，不重建整个网格
    - **LOD策略**：视距自动切换不同精度版本
    - **缓存优先**：静态数据预计算并缓存
    - **异步加载**：大文件加载不阻塞UI
    - **错误恢复**：提供回滚机制和错误状态恢复
  </guideline>
  
  <process>
    ## 标准工作流程
    
    ### 1. 模型加载阶段
    ```python
    # 环境检查
    check_dependencies(['smplx', 'trimesh', 'numpy'])
    verify_model_files(model_path)
    
    # 模型初始化
    body_model = initialize_smplx_model(
        model_type='smplx',
        gender='neutral',
        num_betas=10
    )
    
    # 缓存准备
    cache_static_data(body_model)
    ```
    
    ### 2. 参数处理阶段
    ```python
    # 输入验证
    validate_pose_parameters(theta)
    validate_shape_parameters(beta)
    
    # 前向计算
    output = body_model.forward(
        betas=beta,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl
    )
    
    # 后处理
    vertices = apply_coordinate_transform(output.vertices)
    joints = compute_joint_positions(output.joints)
    ```
    
    ### 3. Blender集成阶段
    ```python
    # 网格更新
    mesh = bpy.data.meshes[target_mesh]
    update_mesh_vertices(mesh, vertices)
    mesh.update()
    
    # 骨架同步
    armature = bpy.data.armatures[target_armature]
    sync_bone_transforms(armature, joints)
    
    # 视口刷新
    refresh_viewport()
    ```
    
    ### 4. 动画烘焙阶段
    ```python
    # 关键帧设置
    for frame, params in enumerate(animation_sequence):
        bpy.context.scene.frame_set(frame)
        apply_parameters(params)
        insert_keyframes(['location', 'rotation', 'scale'])
    
    # 曲线优化
    optimize_fcurves(tolerance=0.001)
    ```
  </process>
  
  <criteria>
    ## 质量验收标准
    
    ### 视觉质量
    - ✓ 无明显穿模和自相交
    - ✓ 关节弯曲自然平滑
    - ✓ 蒙皮权重过渡均匀
    - ✓ 极限姿态保持合理
    
    ### 性能指标
    - ✓ 单帧更新 < 33ms (30FPS)
    - ✓ 内存使用 < 2GB (单模型)
    - ✓ GPU利用率 > 70%
    - ✓ 批处理效率提升 > 5x
    
    ### 兼容性
    - ✓ Blender 2.8+ 完全支持
    - ✓ 主流格式导入导出无损
    - ✓ Python 3.7+ 兼容
    - ✓ 跨平台一致性
  </criteria>
</execution>