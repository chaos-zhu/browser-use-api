// Browser Use WebUI Application
class BrowserUseApp {
    constructor() {
        this.apiBaseUrl = '/api/v1';
        this.currentPage = 1;
        this.tasksPerPage = 12;
        this.currentFilter = '';
        this.tasks = [];
        this.refreshInterval = null;
        this.apiKey = this.loadApiKey();

        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.checkApiStatus();
        await this.loadTasks();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // 表单提交
        document.getElementById('taskForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.createTask();
        });

        // API密钥按钮
        document.getElementById('apiKeyBtn').addEventListener('click', () => {
            this.showApiKeyModal();
        });

        // 刷新按钮
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.loadTasks();
        });

        // 状态过滤器
        document.getElementById('statusFilter').addEventListener('change', (e) => {
            this.currentFilter = e.target.value;
            this.currentPage = 1;
            this.loadTasks();
        });

        // 面板折叠
        document.querySelectorAll('.panel-toggle').forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                const target = e.target.dataset.target;
                const content = document.getElementById(target);
                const icon = e.target;

                content.classList.toggle('collapsed');
                icon.classList.toggle('collapsed');
            });
        });

        // 模态框关闭
        document.querySelectorAll('.modal-close').forEach(closeBtn => {
            closeBtn.addEventListener('click', (e) => {
                const modalId = e.target.dataset.modal || 'taskModal';
                this.closeModal(modalId);
            });
        });

        // 点击模态框外部关闭
        document.getElementById('taskModal').addEventListener('click', (e) => {
            if (e.target.id === 'taskModal') {
                this.closeModal('taskModal');
            }
        });

        document.getElementById('apiKeyModal').addEventListener('click', (e) => {
            if (e.target.id === 'apiKeyModal') {
                this.closeModal('apiKeyModal');
            }
        });

        // API密钥相关事件
        document.getElementById('saveApiKey').addEventListener('click', () => {
            this.saveApiKey();
        });

        document.getElementById('clearApiKey').addEventListener('click', () => {
            this.clearApiKey();
        });

        document.getElementById('testApiKey').addEventListener('click', () => {
            this.testApiKey();
        });

        document.getElementById('toggleApiKeyVisibility').addEventListener('click', () => {
            this.toggleApiKeyVisibility();
        });

        // 清理已完成任务
        document.getElementById('clearCompletedBtn').addEventListener('click', () => {
            this.clearCompletedTasks();
        });

        // ESC键关闭模态框
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
    }

    // API密钥管理方法
    loadApiKey() {
        return localStorage.getItem('browserUseApiKey') || '';
    }

    saveApiKeyToStorage(apiKey) {
        if (apiKey) {
            localStorage.setItem('browserUseApiKey', apiKey);
        } else {
            localStorage.removeItem('browserUseApiKey');
        }
        this.apiKey = apiKey;
    }

    getRequestHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }

        return headers;
    }

    async checkApiStatus() {
        const statusIndicator = document.getElementById('apiStatus');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');

        try {
            const response = await fetch(`${this.apiBaseUrl}/ping`, {
                headers: this.getRequestHeaders()
            });
            if (response.ok) {
                statusDot.classList.remove('error');
                statusText.textContent = '连接正常';
            } else {
                throw new Error('API响应错误');
            }
        } catch (error) {
            statusDot.classList.add('error');
            statusText.textContent = '连接失败';
            this.showNotification('API连接失败', 'error');
        }
    }

    async createTask() {
        const form = document.getElementById('taskForm');
        const formData = new FormData(form);

        const taskData = {
            task: formData.get('task'),
            ai_provider: formData.get('ai_provider') || null,
            save_browser_data: formData.has('save_browser_data'),
            headful: formData.has('headful'),
            use_custom_chrome: formData.has('use_custom_chrome')
        };

        try {
            const response = await fetch(`${this.apiBaseUrl}/run-task`, {
                method: 'POST',
                headers: this.getRequestHeaders(),
                body: JSON.stringify(taskData)
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(`任务已创建: ${result.id}`, 'success');
                form.reset();
                await this.loadTasks();
            } else {
                const error = await response.json();
                throw new Error(error.detail || '创建任务失败');
            }
        } catch (error) {
            this.showNotification(`创建任务失败: ${error.message}`, 'error');
        }
    }

    async loadTasks() {
        try {
            const params = new URLSearchParams({
                page: this.currentPage,
                per_page: this.tasksPerPage
            });

            const response = await fetch(`${this.apiBaseUrl}/list-tasks?${params}`, {
                headers: this.getRequestHeaders()
            });
            if (response.ok) {
                const data = await response.json();
                this.tasks = data.tasks || [];
                this.renderTasks();
                this.renderPagination(data.total_pages || 1, data.total_tasks || 0);
                this.updateTaskCount(data.total_tasks || 0);
            } else {
                throw new Error('加载任务列表失败');
            }
        } catch (error) {
            this.showNotification(`加载任务失败: ${error.message}`, 'error');
            this.renderEmptyState();
        }
    }

    renderTasks() {
        const taskGrid = document.getElementById('taskGrid');

        if (this.tasks.length === 0) {
            this.renderEmptyState();
            return;
        }

        // 过滤任务
        let filteredTasks = this.tasks;
        if (this.currentFilter) {
            filteredTasks = this.tasks.filter(task => task.status === this.currentFilter);
        }

        const tasksHtml = filteredTasks.map(task => this.renderTaskCard(task)).join('');
        taskGrid.innerHTML = tasksHtml;

        // 添加事件监听器
        this.attachTaskEventListeners();
    }

    renderTaskCard(task) {
        const createdAt = new Date(task.created_at).toLocaleString('zh-CN');
        const finishedAt = task.finished_at ? new Date(task.finished_at).toLocaleString('zh-CN') : '未完成';

        return `
            <div class="task-card status-${task.status}" data-task-id="${task.id}">
                <div class="task-header">
                    <div class="task-id">${task.id.substring(0, 8)}...</div>
                    <div class="task-status status-${task.status}">${this.getStatusText(task.status)}</div>
                </div>
                <div class="task-description">${task.task}</div>
                <div class="task-meta">
                    <span><i class="fas fa-robot"></i> ${task.ai_provider || '默认'}</span>
                    <span><i class="fas fa-clock"></i> ${createdAt}</span>
                </div>
                <div class="task-actions">
                    <button class="btn btn-small btn-secondary view-task" data-task-id="${task.id}">
                        <i class="fas fa-eye"></i> 查看
                    </button>
                    ${this.renderTaskActionButtons(task)}
                </div>
            </div>
        `;
    }

    renderTaskActionButtons(task) {
        const buttons = [];

        if (task.status === 'running') {
            buttons.push(`
                <button class="btn btn-small btn-warning pause-task" data-task-id="${task.id}">
                    <i class="fas fa-pause"></i> 暂停
                </button>
                <button class="btn btn-small btn-danger stop-task" data-task-id="${task.id}">
                    <i class="fas fa-stop"></i> 停止
                </button>
            `);
        } else if (task.status === 'paused') {
            buttons.push(`
                <button class="btn btn-small btn-success resume-task" data-task-id="${task.id}">
                    <i class="fas fa-play"></i> 恢复
                </button>
                <button class="btn btn-small btn-danger stop-task" data-task-id="${task.id}">
                    <i class="fas fa-stop"></i> 停止
                </button>
            `);
        }

        if (['finished', 'failed', 'stopped'].includes(task.status)) {
            buttons.push(`
                <button class="btn btn-small btn-secondary view-media" data-task-id="${task.id}">
                    <i class="fas fa-images"></i> 媒体
                </button>
            `);
        }

        if (task.live_url) {
            buttons.push(`
                <a href="${task.live_url}" target="_blank" class="btn btn-small btn-secondary">
                    <i class="fas fa-external-link-alt"></i> 实时
                </a>
            `);
        }

        return buttons.join('');
    }

    attachTaskEventListeners() {
        // 查看任务详情
        document.querySelectorAll('.view-task').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('[data-task-id]').dataset.taskId;
                this.viewTaskDetails(taskId);
            });
        });

        // 暂停任务
        document.querySelectorAll('.pause-task').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('[data-task-id]').dataset.taskId;
                this.pauseTask(taskId);
            });
        });

        // 恢复任务
        document.querySelectorAll('.resume-task').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('[data-task-id]').dataset.taskId;
                this.resumeTask(taskId);
            });
        });

        // 停止任务
        document.querySelectorAll('.stop-task').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('[data-task-id]').dataset.taskId;
                if (confirm('确定要停止这个任务吗？此操作无法撤销。')) {
                    this.stopTask(taskId);
                }
            });
        });

        // 查看媒体
        document.querySelectorAll('.view-media').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('[data-task-id]').dataset.taskId;
                this.viewTaskMedia(taskId);
            });
        });
    }

    renderEmptyState() {
        const taskGrid = document.getElementById('taskGrid');
        taskGrid.innerHTML = `
            <div class="loading-placeholder">
                <i class="fas fa-inbox"></i>
                <p>暂无任务</p>
            </div>
        `;
    }

    renderPagination(totalPages, totalTasks) {
        const pagination = document.getElementById('pagination');

        if (totalPages <= 1) {
            pagination.innerHTML = '';
            return;
        }

        let paginationHtml = '';

        // 上一页按钮
        paginationHtml += `
            <button ${this.currentPage === 1 ? 'disabled' : ''} onclick="app.goToPage(${this.currentPage - 1})">
                <i class="fas fa-chevron-left"></i>
            </button>
        `;

        // 页码按钮
        for (let i = 1; i <= totalPages; i++) {
            if (i === this.currentPage || i === 1 || i === totalPages ||
                (i >= this.currentPage - 1 && i <= this.currentPage + 1)) {
                paginationHtml += `
                    <button class="${i === this.currentPage ? 'active' : ''}" onclick="app.goToPage(${i})">
                        ${i}
                    </button>
                `;
            } else if (i === this.currentPage - 2 || i === this.currentPage + 2) {
                paginationHtml += '<span>...</span>';
            }
        }

        // 下一页按钮
        paginationHtml += `
            <button ${this.currentPage === totalPages ? 'disabled' : ''} onclick="app.goToPage(${this.currentPage + 1})">
                <i class="fas fa-chevron-right"></i>
            </button>
        `;

        pagination.innerHTML = paginationHtml;
    }

    goToPage(page) {
        this.currentPage = page;
        this.loadTasks();
    }

    updateTaskCount(count) {
        document.getElementById('taskCount').textContent = `任务总数: ${count}`;
    }

    getStatusText(status) {
        const statusMap = {
            'created': '已创建',
            'running': '运行中',
            'finished': '已完成',
            'failed': '失败',
            'stopped': '已停止',
            'paused': '已暂停',
            'stopping': '停止中'
        };
        return statusMap[status] || status;
    }

    startAutoRefresh() {
        // 每5秒自动刷新任务列表
        this.refreshInterval = setInterval(() => {
            this.loadTasks();
        }, 5000);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        const notifications = document.getElementById('notifications');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;

        notification.innerHTML = `
            <div class="notification-header">
                <div class="notification-title">${this.getNotificationTitle(type)}</div>
                <button class="notification-close">&times;</button>
            </div>
            <div class="notification-message">${message}</div>
        `;

        notifications.appendChild(notification);

        // 关闭按钮事件
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });

        // 自动关闭
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, duration);
    }

    getNotificationTitle(type) {
        const titles = {
            'success': '成功',
            'error': '错误',
            'warning': '警告',
            'info': '信息'
        };
        return titles[type] || '通知';
    }

    closeModal(modalId = 'taskModal') {
        document.getElementById(modalId).style.display = 'none';
    }

    closeAllModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    }

    showApiKeyModal() {
        const modal = document.getElementById('apiKeyModal');
        const input = document.getElementById('apiKeyInput');
        const status = document.getElementById('apiKeyStatus');

        // 加载当前API密钥
        input.value = this.apiKey;
        status.style.display = 'none';

        modal.style.display = 'block';
    }

    saveApiKey() {
        const input = document.getElementById('apiKeyInput');
        const apiKey = input.value.trim();

        this.saveApiKeyToStorage(apiKey);
        this.showApiKeyStatus('API密钥已保存', 'success');

        // 重新检查API状态
        this.checkApiStatus();

        setTimeout(() => {
            this.closeModal('apiKeyModal');
        }, 1500);
    }

    clearApiKey() {
        if (confirm('确定要清除API密钥吗？')) {
            document.getElementById('apiKeyInput').value = '';
            this.saveApiKeyToStorage('');
            this.showApiKeyStatus('API密钥已清除', 'info');
            this.checkApiStatus();
        }
    }

    async testApiKey() {
        const input = document.getElementById('apiKeyInput');
        const testKey = input.value.trim();

        try {
            const headers = {
                'Content-Type': 'application/json'
            };

            if (testKey) {
                headers['X-API-Key'] = testKey;
            }

            const response = await fetch(`${this.apiBaseUrl}/ping`, { headers });

            if (response.ok) {
                this.showApiKeyStatus('API密钥有效，连接成功！', 'success');
            } else {
                this.showApiKeyStatus('API密钥无效或连接失败', 'error');
            }
        } catch (error) {
            this.showApiKeyStatus(`连接测试失败: ${error.message}`, 'error');
        }
    }

    toggleApiKeyVisibility() {
        const input = document.getElementById('apiKeyInput');
        const button = document.getElementById('toggleApiKeyVisibility');
        const icon = button.querySelector('i');

        if (input.type === 'password') {
            input.type = 'text';
            icon.className = 'fas fa-eye-slash';
        } else {
            input.type = 'password';
            icon.className = 'fas fa-eye';
        }
    }

    showApiKeyStatus(message, type) {
        const status = document.getElementById('apiKeyStatus');
        status.textContent = message;
        status.className = `api-key-status ${type}`;
        status.style.display = 'block';
    }

    async viewTaskDetails(taskId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/task/${taskId}`, {
                headers: this.getRequestHeaders()
            });
            if (response.ok) {
                const task = await response.json();
                this.renderTaskModal(task);
            } else {
                throw new Error('获取任务详情失败');
            }
        } catch (error) {
            this.showNotification(`获取任务详情失败: ${error.message}`, 'error');
        }
    }

    renderTaskModal(task) {
        const modal = document.getElementById('taskModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');

        modalTitle.textContent = `任务详情 - ${task.id.substring(0, 8)}...`;

        const createdAt = new Date(task.created_at).toLocaleString('zh-CN');
        const finishedAt = task.finished_at ? new Date(task.finished_at).toLocaleString('zh-CN') : '未完成';

        modalBody.innerHTML = `
            <div class="task-detail">
                <div class="detail-section">
                    <h4><i class="fas fa-info-circle"></i> 基本信息</h4>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-label">任务ID</div>
                            <div class="detail-value">${task.id}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">状态</div>
                            <div class="detail-value">
                                <span class="task-status status-${task.status}">${this.getStatusText(task.status)}</span>
                            </div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">AI提供商</div>
                            <div class="detail-value">${task.ai_provider || '默认'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">创建时间</div>
                            <div class="detail-value">${createdAt}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">完成时间</div>
                            <div class="detail-value">${finishedAt}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">保存浏览器数据</div>
                            <div class="detail-value">${task.save_browser_data ? '是' : '否'}</div>
                        </div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">任务描述</div>
                        <div class="detail-value" style="white-space: pre-wrap;">${task.task}</div>
                    </div>
                </div>

                ${task.output ? `
                    <div class="detail-section">
                        <h4><i class="fas fa-check-circle"></i> 执行结果</h4>
                        <div class="detail-value" style="white-space: pre-wrap; max-height: 200px; overflow-y: auto;">${task.output}</div>
                    </div>
                ` : ''}

                ${task.error ? `
                    <div class="detail-section" style="border-left-color: #ef4444;">
                        <h4><i class="fas fa-exclamation-triangle"></i> 错误信息</h4>
                        <div class="detail-value" style="color: #dc2626; white-space: pre-wrap;">${task.error}</div>
                    </div>
                ` : ''}

                ${task.steps && task.steps.length > 0 ? `
                    <div class="detail-section">
                        <h4><i class="fas fa-list-ol"></i> 执行步骤 (${task.steps.length})</h4>
                        <div class="steps-list">
                            ${task.steps.map(step => `
                                <div class="step-item">
                                    <div class="step-header">
                                        <div class="step-number">步骤 ${step.step}</div>
                                        <div class="step-time">${step.timestamp ? new Date(step.timestamp).toLocaleString('zh-CN') : ''}</div>
                                    </div>
                                    <div class="step-content">
                                        <strong>目标:</strong> ${step.next_goal || '无'}<br>
                                        <strong>评估:</strong> ${step.evaluation_previous_goal || '无'}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                <div class="detail-section">
                    <h4><i class="fas fa-cogs"></i> 操作</h4>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        ${this.renderModalActionButtons(task)}
                        ${task.live_url ? `
                            <a href="${task.live_url}" target="_blank" class="btn btn-secondary">
                                <i class="fas fa-external-link-alt"></i> 实时监控
                            </a>
                        ` : ''}
                        <button class="btn btn-secondary" onclick="app.loadTaskMedia('${task.id}')">
                            <i class="fas fa-images"></i> 查看媒体
                        </button>
                    </div>
                </div>
            </div>
        `;

        modal.style.display = 'block';
    }

    renderModalActionButtons(task) {
        const buttons = [];

        if (task.status === 'running') {
            buttons.push(`
                <button class="btn btn-warning" onclick="app.pauseTask('${task.id}')">
                    <i class="fas fa-pause"></i> 暂停任务
                </button>
                <button class="btn btn-danger" onclick="app.stopTask('${task.id}')">
                    <i class="fas fa-stop"></i> 停止任务
                </button>
            `);
        } else if (task.status === 'paused') {
            buttons.push(`
                <button class="btn btn-success" onclick="app.resumeTask('${task.id}')">
                    <i class="fas fa-play"></i> 恢复任务
                </button>
                <button class="btn btn-danger" onclick="app.stopTask('${task.id}')">
                    <i class="fas fa-stop"></i> 停止任务
                </button>
            `);
        }

        return buttons.join('');
    }

    async pauseTask(taskId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/pause-task/${taskId}`, {
                method: 'PUT',
                headers: this.getRequestHeaders()
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message, 'success');
                await this.loadTasks();
                this.closeModal();
            } else {
                const error = await response.json();
                throw new Error(error.detail || '暂停任务失败');
            }
        } catch (error) {
            this.showNotification(`暂停任务失败: ${error.message}`, 'error');
        }
    }

    async resumeTask(taskId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/resume-task/${taskId}`, {
                method: 'PUT',
                headers: this.getRequestHeaders()
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message, 'success');
                await this.loadTasks();
                this.closeModal();
            } else {
                const error = await response.json();
                throw new Error(error.detail || '恢复任务失败');
            }
        } catch (error) {
            this.showNotification(`恢复任务失败: ${error.message}`, 'error');
        }
    }

    async stopTask(taskId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/stop-task/${taskId}`, {
                method: 'PUT',
                headers: this.getRequestHeaders()
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message, 'success');
                await this.loadTasks();
                this.closeModal();
            } else {
                const error = await response.json();
                throw new Error(error.detail || '停止任务失败');
            }
        } catch (error) {
            this.showNotification(`停止任务失败: ${error.message}`, 'error');
        }
    }

    async loadTaskMedia(taskId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/task/${taskId}/media/list`, {
                headers: this.getRequestHeaders()
            });
            if (response.ok) {
                const data = await response.json();
                this.renderMediaModal(taskId, data.media);
            } else {
                throw new Error('获取媒体列表失败');
            }
        } catch (error) {
            this.showNotification(`获取媒体失败: ${error.message}`, 'error');
        }
    }

    renderMediaModal(taskId, mediaList) {
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');

        modalTitle.textContent = `任务媒体 - ${taskId.substring(0, 8)}...`;

        if (mediaList.length === 0) {
            modalBody.innerHTML = `
                <div class="detail-section">
                    <h4><i class="fas fa-images"></i> 媒体文件</h4>
                    <p style="text-align: center; color: #64748b; padding: 40px;">
                        <i class="fas fa-inbox" style="font-size: 2rem; margin-bottom: 10px; display: block;"></i>
                        暂无媒体文件
                    </p>
                </div>
            `;
        } else {
            modalBody.innerHTML = `
                <div class="detail-section">
                    <h4><i class="fas fa-images"></i> 媒体文件 (${mediaList.length})</h4>
                    <div class="media-grid">
                        ${mediaList.map(media => `
                            <div class="media-item">
                                ${media.type === 'screenshot' ? `
                                    <img src="${media.url}" alt="${media.filename}" class="media-preview"
                                         onclick="window.open('${media.url}', '_blank')">
                                ` : `
                                    <div class="media-preview" style="display: flex; align-items: center; justify-content: center; background: #f1f5f9;">
                                        <i class="fas fa-file" style="font-size: 2rem; color: #64748b;"></i>
                                    </div>
                                `}
                                <div class="media-info">
                                    <div><strong>${media.filename}</strong></div>
                                    <div>类型: ${media.type}</div>
                                    <div>大小: ${this.formatFileSize(media.size_bytes)}</div>
                                    <div>创建: ${new Date(media.created_at).toLocaleString('zh-CN')}</div>
                                    <div style="margin-top: 8px;">
                                        <a href="${media.url}" target="_blank" class="btn btn-small btn-secondary">
                                            <i class="fas fa-eye"></i> 查看
                                        </a>
                                        <a href="${media.url}?download=true" class="btn btn-small btn-secondary">
                                            <i class="fas fa-download"></i> 下载
                                        </a>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async viewTaskMedia(taskId) {
        await this.loadTaskMedia(taskId);
    }

    async clearCompletedTasks() {
        if (!confirm('确定要清理所有已完成的任务吗？此操作无法撤销。')) {
            return;
        }

        // 这里可以实现清理已完成任务的逻辑
        // 由于API没有提供批量删除功能，这里只是一个占位符
        this.showNotification('清理功能待实现', 'info');
    }
}

// 初始化应用
const app = new BrowserUseApp();
