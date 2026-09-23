<!DOCTYPE html >
<html lang = "ja" >

<head >
    <meta charset = "UTF-8" >
    <link rel = "stylesheet" href = "style.css" >
</head >

<body style = "padding: 10px;" >

    <!-- 稲妻描画用キャンバス - ->
    <canvas id = "lightning-canvas" > </canvas >

    <!-- 基本習慣 & Streak表示 - ->
    <div class = "card" >
        <div style = "display: flex; justify-content: space-between; align-items: center;" >
            <h2 > 🔥 TODAY'S STATUS</h2>
            <div id="streak-badge"
                style="font-weight: 800; font-size: 0.95rem; padding: 6px 12px; border-radius: 6px; background: #0f111a; border: 1px solid var(--border-color);">
                STREAK: <span id="streak-count" style="color: var(--accent-blue);">0</span> DAYS
            </div>
        </div>

        <form id="daily-form">
            <div style="display: flex; gap: 20px; align-items: flex-end; flex-wrap: wrap; margin-top: 10px;">
                <div style="flex: 1; min-width: 180px;" class="form-group">
                    <label>学校到着時間</label>
                    <input type="time" id="arrival_time">
                </div>

                <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                    <label class="cyber-switch">
                        <input type="checkbox" id="no_fap" checked>
                        <span class="slider fap">NO FAP</span>
                    </label>

                    <label class="cyber-switch">
                        <input type="checkbox" id="no_porn" checked>
                        <span class="slider porn">NO PORN</span>
                    </label>
                </div>
            </div>
            <button type="submit" class="btn-primary" style="margin-top: 10px;">Status 保存</button>
        </form>
    </div>

    <!-- 集中タイマー -->
    <div class="card">
        <h2>⏱️ FOCUS SESSION</h2>
        <div class="form-group">
            <label>集中項目</label>
            <select id="category_select"></select>
        </div>

        <div style="text-align: center; margin: 20px 0;">
            <div id="timer-display" style="font-size: 3rem; font-weight: bold; color: var(--accent-green);">00:00:00</div>
            
            <!-- ボタン描画領域（CSS重なり完全防止設定） -->
            <div id="button-container"></div>
        </div>

        <hr style="border-color: var(--border-color); margin: 20px 0;">

        <div style="display: flex; gap: 10px; align-items: flex-end;">
            <div style="flex: 1;">
                <label>手動入力 (分)</label>
                <input type="number" id="manual_minutes" placeholder="60">
            </div>
            <button class="btn-primary" onclick="saveManualSession()">直接追加</button>
        </div>
    </div>

    <!-- 本日の記録一覧 & 合計時間 -->
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2>📊 TODAY'S LOGS</h2>
            <div class="total-time-badge">
                TOTAL: <span id="total-focus-time" style="color: #fff; font-size: 1.1rem;">0</span> 分
            </div>
        </div>
        <div id="session-list" style="margin-top: 15px;"></div>
    </div>

    <script>
        const API_BASE = "http://localhost:3001";
        
        function getTodayString() {
            const nowObj = new Date();
            const year = nowObj.getFullYear();
            const month = String(nowObj.getMonth() + 1).padStart(2, '0');
            const day = String(nowObj.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
        }

        const TODAY = getTodayString();

        let timerInterval = null;
        let startTime = null;
        let accumulatedTime = 0;
        let isPaused = false;

        document.addEventListener("DOMContentLoaded", () => {
            checkDateChangeAndReset();
            loadCategories();
            loadTodayData();
            calculateStreak();
            restoreTimerState();
        });

        function checkDateChangeAndReset() {
            const savedDate = localStorage.getItem('timer_saved_date');
            if (savedDate && savedDate !== TODAY) {
                localStorage.removeItem('timer_start_time');
                localStorage.removeItem('timer_accumulated_time');
                localStorage.removeItem('timer_is_paused');
            }
            localStorage.setItem('timer_saved_date', TODAY);
        }

        // --- DOM生成＆絶対配置(重なり)強力解除ロジック ---
        function renderButtons() {
            const container = document.getElementById('button-container');

            // 親コンテナの横並び配置を強制
            container.style.cssText = "display: flex !important; justify-content: center !important; align-items: center !important; gap: 15px !important; flex-wrap: wrap !important; width: 100% !important; position: static !important; margin-top: 15px !important;";

            if (!startTime) {
                // 【初期状態（STARTのみ）】
                container.innerHTML = `
                    <button class="btn-primary" onclick="toggleTimer()"
                        style="position: static !important; font-size: 1.1rem; padding: 12px 30px; min-width: 140px; margin: 0 !important; display: inline-block !important;">START</button>
                `;
            } else {
                // 【計測中（STOP & SAVE + PAUSE/RESUME 横並び）】
                const pauseText = isPaused ? "RESUME" : "PAUSE";
                const pauseBg = isPaused ? "var(--accent-blue)" : "#ff9800";

                container.innerHTML = `
                    <button class="btn-primary" onclick="toggleTimer()"
                        style="position: static !important; font-size: 1.1rem; padding: 12px 30px; min-width: 140px; margin: 0 !important; display: inline-block !important; background: var(--danger-color) !important;">STOP & SAVE</button>
                    <button class="btn-primary" onclick="togglePause()"
                        style="position: static !important; font-size: 1.1rem; padding: 12px 30px; min-width: 140px; margin: 0 !important; display: inline-block !important; background: ${pauseBg} !important;">${pauseText}</button>
                `;
            }
        }

        function restoreTimerState() {
            const savedStartTime = localStorage.getItem('timer_start_time');
            const savedAccumulated = localStorage.getItem('timer_accumulated_time');
            const savedIsPaused = localStorage.getItem('timer_is_paused');

            if (savedAccumulated) {
                accumulatedTime = parseInt(savedAccumulated, 10) || 0;
            }

            if (savedStartTime) {
                startTime = parseInt(savedStartTime, 10);
                isPaused = (savedIsPaused === 'true');

                if (!isPaused) {
                    timerInterval = setInterval(updateTimerDisplay, 500);
                } else {
                    updateTimerDisplay();
                }
            } else {
                startTime = null;
                accumulatedTime = 0;
                isPaused = false;
            }

            renderButtons();
        }

        function toggleTimer() {
            if (!startTime) {
                // START
                startTime = Date.now();
                accumulatedTime = 0;
                isPaused = false;

                localStorage.setItem('timer_start_time', startTime);
                localStorage.setItem('timer_accumulated_time', 0);
                localStorage.setItem('timer_is_paused', 'false');
                localStorage.setItem('timer_saved_date', TODAY);

                timerInterval = setInterval(updateTimerDisplay, 500);
            } else {
                // STOP & SAVE
                if (timerInterval) clearInterval(timerInterval);
                timerInterval = null;

                let totalMs = accumulatedTime;
                if (!isPaused && startTime) {
                    totalMs += (Date.now() - startTime);
                }

                const totalSeconds = Math.floor(totalMs / 1000);
                const durationMinutes = Math.round(totalSeconds / 60) || (totalSeconds > 0 ? 1 : 0);

                if (durationMinutes > 0) {
                    saveSession(durationMinutes);
                }

                startTime = null;
                accumulatedTime = 0;
                isPaused = false;

                localStorage.removeItem('timer_start_time');
                localStorage.removeItem('timer_accumulated_time');
                localStorage.removeItem('timer_is_paused');

                document.getElementById('timer-display').textContent = "00:00:00";
            }

            renderButtons();
        }

        function togglePause() {
            if (!isPaused) {
                // PAUSE
                clearInterval(timerInterval);
                timerInterval = null;

                accumulatedTime += (Date.now() - startTime);
                isPaused = true;

                localStorage.setItem('timer_accumulated_time', accumulatedTime);
                localStorage.setItem('timer_is_paused', 'true');

                updateTimerDisplay();
            } else {
                // RESUME
                startTime = Date.now();
                isPaused = false;

                localStorage.setItem('timer_start_time', startTime);
                localStorage.setItem('timer_is_paused', 'false');

                timerInterval = setInterval(updateTimerDisplay, 500);
            }

            renderButtons();
        }

        function updateTimerDisplay() {
            let totalMs = accumulatedTime;
            if (!isPaused && startTime) {
                totalMs += (Date.now() - startTime);
            }

            const secondsElapsed = Math.floor(totalMs / 1000);
            const h = String(Math.floor(secondsElapsed / 3600)).padStart(2, '0');
            const m = String(Math.floor((secondsElapsed % 3600) / 60)).padStart(2, '0');
            const s = String(secondsElapsed % 60).padStart(2, '0');

            document.getElementById('timer-display').textContent = `${h}:${m}:${s}`;
        }

        async function loadCategories() {
            try {
                const res = await fetch(`${API_BASE}/focus_categories`);
                const data = await res.json();
                const select = document.getElementById('category_select');
                select.innerHTML = data.map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            } catch (err) {
                console.error("カテゴリ読み込みエラー:", err);
            }
        }

        async function loadTodayData() {
            try {
                const dailyRes = await fetch(`${API_BASE}/daily_records?record_date=eq.${TODAY}`);
                const dailyData = await dailyRes.json();
                if (dailyData.length > 0) {
                    document.getElementById('arrival_time').value = dailyData[0].arrival_time || '';
                    document.getElementById('no_fap').checked = dailyData[0].no_fap;
                    document.getElementById('no_porn').checked = dailyData[0].no_porn;
                }

                const sessionRes = await fetch(`${API_BASE}/focus_sessions?record_date=eq.${TODAY}`);
                const sessions = await sessionRes.json();

                const totalMinutes = sessions.reduce((sum, s) => sum + (s.duration_minutes || 0), 0);
                document.getElementById('total-focus-time').textContent = totalMinutes;

                const list = document.getElementById('session-list');
                list.innerHTML = sessions.map(s => `
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding: 8px 0;">
                        <span>${s.category_name}</span>
                        <div>
                            <strong style="color: var(--accent-blue);">${s.duration_minutes} 分</strong>
                            <button onclick="deleteSession(${s.id})" style="background: none; border: none; color: var(--danger-color); cursor: pointer; margin-left: 10px;">削除</button>
                        </div>
                    </div>
                `).join('') || '<div style="color: var(--text-muted);">本日の記録はまだありません</div>';
            } catch (err) {
                console.error("本日データ読み込みエラー:", err);
            }
        }

        async function calculateStreak() {
            try {
                const res = await fetch(`${API_BASE}/daily_records?order=record_date.desc`);
                const records = await res.json();

                let streak = 0;
                for (let r of records) {
                    if (r.no_fap && r.no_porn) {
                        streak++;
                    } else {
                        break;
                    }
                }

                document.getElementById('streak-count').textContent = streak;
                applyVisualEffects(streak);
            } catch (err) {
                console.error("Streak計算エラー:", err);
            }
        }

        function applyVisualEffects(streak) {
            document.body.classList.remove('aura-effect', 'grid-effect', 'lightning-active');
            const canvas = document.getElementById('lightning-canvas');
            canvas.style.display = 'none';

            if (streak >= 30) {
                document.body.classList.add('aura-effect', 'grid-effect', 'lightning-active');
                canvas.style.display = 'block';
                startLightningEffect();
            } else if (streak >= 7) {
                document.body.classList.add('aura-effect', 'grid-effect');
            } else if (streak >= 3) {
                document.body.classList.add('aura-effect');
            }
        }

        function startLightningEffect() {
            const canvas = document.getElementById('lightning-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;

            function drawBolt() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (Math.random() > 0.85) {
                    ctx.beginPath();
                    let x = Math.random() * canvas.width;
                    let y = 0;
                    ctx.moveTo(x, y);

                    while (y < canvas.height) {
                        x += (Math.random() - 0.5) * 60;
                        y += Math.random() * 40;
                        ctx.lineTo(x, y);
                    }

                    ctx.strokeStyle = Math.random() > 0.5 ? '#00d2ff' : '#ffffff';
                    ctx.lineWidth = Math.random() * 3 + 1;
                    ctx.shadowBlur = 20;
                    ctx.shadowColor = '#00d2ff';
                    ctx.stroke();
                }
                requestAnimationFrame(drawBolt);
            }
            drawBolt();
        }

        async function saveManualSession() {
            const mins = parseFloat(document.getElementById('manual_minutes').value);
            if (mins > 0) {
                await saveSession(mins);
                document.getElementById('manual_minutes').value = '';
            }
        }

        async function saveSession(duration_minutes) {
            const category_name = document.getElementById('category_select').value;
            await fetch(`${API_BASE}/daily_records`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Prefer': 'resolution=ignore-duplicates' },
                body: JSON.stringify({ record_date: TODAY })
            });
            await fetch(`${API_BASE}/focus_sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ record_date: TODAY, category_name, duration_minutes })
            });
            loadTodayData();
        }

        async function deleteSession(id) {
            await fetch(`${API_BASE}/focus_sessions?id=eq.${id}`, { method: 'DELETE' });
            loadTodayData();
        }

        document.getElementById('daily-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            await fetch(`${API_BASE}/daily_records`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Prefer': 'resolution=merge-duplicates' },
                body: JSON.stringify({
                    record_date: TODAY,
                    arrival_time: document.getElementById('arrival_time').value || null,
                    no_fap: document.getElementById('no_fap').checked,
                    no_porn: document.getElementById('no_porn').checked
                })
            });
            alert('保存しました！');
            calculateStreak();
        });
    </script>
</body>

</html>