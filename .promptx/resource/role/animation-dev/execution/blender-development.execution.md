<execution>
  <constraint>
    ## Blender开发约束
    - **API版本**：必须检查bpy.app.version兼容性
    - **上下文要求**：某些操作必须在特定context下执行
    - **线程限制**：Blender Python不支持真正的多线程
    - **模态操作**：modal operator会阻塞其他操作
    - **注册机制**：类必须继承自bpy.types并正确注册
  </constraint>
  
  <rule>
    ## Blender插件规范
    - **命名规则**：类名必须包含有效前缀(如MESH_OT_)
    - **bl_info必需**：版本、作者、类别等元信息
    - **属性注册**：使用bpy.props而非Python原生类型
    - **撤销支持**：操作必须支持undo/redo
    - **国际化**：UI文本使用翻译系统
  </rule>
  
  <guideline>
    ## 插件开发最佳实践
    - **操作原子化**：每个operator完成单一功能
    - **UI响应式**：使用draw回调动态更新面板
    - **错误处理**：使用report()而非print()
    - **性能优化**：使用bmesh进行大量编辑操作
    - **用户友好**：提供清晰的工具提示和文档
  </guideline>
  
  <process>
    ## Blender插件开发流程
    
    ### 1. 插件结构设计
    ```python
    bl_info = {
        "name": "SMPL-X Toolbox",
        "author": "Animation Dev",
        "version": (1, 0, 0),
        "blender": (2, 80, 0),
        "category": "Animation",
    }
    
    # 模块组织
    ├── __init__.py      # 插件入口
    ├── operators/       # 操作类
    ├── panels/          # UI面板
    ├── properties/      # 属性定义
    └── utils/          # 工具函数
    ```
    
    ### 2. 操作器实现
    ```python
    class SMPLX_OT_load_model(bpy.types.Operator):
        bl_idname = "smplx.load_model"
        bl_label = "Load SMPL-X Model"
        bl_options = {'REGISTER', 'UNDO'}
        
        def execute(self, context):
            # 核心逻辑
            model = load_smplx_model()
            create_mesh_from_model(model)
            return {'FINISHED'}
    ```
    
    ### 3. UI面板集成
    ```python
    class SMPLX_PT_main_panel(bpy.types.Panel):
        bl_label = "SMPL-X Controls"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "SMPL-X"
        
        def draw(self, context):
            layout = self.layout
            layout.operator("smplx.load_model")
            layout.prop(context.scene, "smplx_shape_params")
    ```
    
    ### 4. 属性系统
    ```python
    def register_properties():
        bpy.types.Scene.smplx_model_path = bpy.props.StringProperty(
            name="Model Path",
            subtype='FILE_PATH'
        )
        bpy.types.Scene.smplx_shape_params = bpy.props.FloatVectorProperty(
            name="Shape",
            size=10,
            min=-5, max=5
        )
    ```
    
    ### 5. 性能优化技巧
    ```python
    # 使用bmesh批量编辑
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # 批量操作...
    bm.to_mesh(mesh)
    bm.free()
    
    # 延迟视口更新
    with context.temp_override(window=window):
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    ```
  </process>
  
  <criteria>
    ## 插件质量标准
    
    ### 功能完整性
    - ✓ 核心功能全部实现
    - ✓ 错误处理完善
    - ✓ 支持撤销/重做
    - ✓ 参数验证严格
    
    ### 用户体验
    - ✓ UI布局合理直观
    - ✓ 操作响应迅速
    - ✓ 提示信息清晰
    - ✓ 快捷键配置合理
    
    ### 代码质量
    - ✓ 遵循PEP 8规范
    - ✓ 模块划分清晰
    - ✓ 注释完整准确
    - ✓ 无内存泄漏
  </criteria>
</execution>