"""
Streamlit + Supabase AI-enabled Multi-User Dashboard
---------------------------------------------------
وصف:
- نظام بمستخدمين من نوعين: Admin و User
- Admin يمكنه إضافة مستخدمين (Users) وتعيينهم مهامًا
- كل User يرى صفحته الخاصة فقط
- تخزين البيانات على Supabase (Postgres)
- ألوان عرض المهام: todo=أحمر، inprogress=أصفر، done=أخضر.
  إذا مضى تاريخ الإنجاز ولم تُنجز المهمة تظهر كـ "نصف أحمر/نصف أصفر" (عرض بالـ HTML)
- ميزات AI (اختيارية): تلخيص المهام، تحليل الأداء، تنبيهات، وتوقع مشاكل باستخدام نموذج بسيط

متطلبات تنصيب (pip):
- pip install streamlit supabase-python python-dotenv openai pandas plotly scikit-learn

المتغيرات البيئية المطلوبة (env vars):
- SUPABASE_URL = https://xyz.supabase.co
- SUPABASE_ANON_KEY = <your-supabase-anon-key>
- SUPABASE_SERVICE_ROLE_KEY = <optional - required to create users programmatically>
- OPENAI_API_KEY = <optional - for AI summary & analysis using OpenAI>

ملاحظات أمان:
- لا تضع SERVICE_ROLE_KEY في كود عام. احفظه في متغير بيئي على الخادم.
- إذا اخترت Streamlit Cloud أو استضافة عامة، ضع SERVICE_ROLE_KEY فقط في بيئة خاصة للسرفر.

إنشاء الجداول في Supabase (SQL) - شغّل هذه الاستعلامات في SQL Editor داخل Supabase:

-- users table (profiles) عادةً Supabase يولد جدول profiles مع auth; لكن إن أردت جدول مخصص:
CREATE TABLE IF NOT EXISTS profiles (
  id uuid PRIMARY KEY,
  email text UNIQUE,
  full_name text,
  role text DEFAULT 'user', -- 'admin' أو 'user'
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tasks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text NOT NULL,
  description text,
  owner uuid REFERENCES profiles(id) ON DELETE SET NULL,
  status text DEFAULT 'todo', -- 'todo', 'inprogress', 'done'
  created_at timestamptz DEFAULT now(),
  due_date date,
  completed_at timestamptz,
  metadata jsonb
);

-- Index لتحسين الأداء
CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks (owner);


التعليمات السريعة لتشغيل التطبيق:
1) جهز متغيرات البيئة.
2) شغّل: streamlit run streamlit_supabase_dashboard.py

------------------------------------------------------------------
الكود:
------------------------------------------------------------------
"""

import os
import streamlit as st
from supabase import create_client, Client
from datetime import datetime, date
import pandas as pd
import plotly.express as px
import json
from typing import Optional

# Optional AI
try:
    import openai
except Exception:
    openai = None

# Optional ML
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except Exception:
    LogisticRegression = None

# ---------------------- Configuration ----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # required for admin user creation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Environment variables SUPABASE_URL and SUPABASE_ANON_KEY must be set.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------------- Helpers: DB & Auth ----------------------

def get_profile_by_email(email: str) -> Optional[dict]:
    res = supabase.table('profiles').select('*').eq('email', email).limit(1).execute()
    data = res.data
    return data[0] if data else None

def get_profile_by_id(uid: str) -> Optional[dict]:
    res = supabase.table('profiles').select('*').eq('id', uid).limit(1).execute()
    data = res.data
    return data[0] if data else None

def list_users() -> list:
    res = supabase.table('profiles').select('*').order('created_at', desc=False).execute()
    return res.data or []

# create user - requires service role key; we will call Supabase Auth REST endpoint if SERVICE_ROLE_KEY is provided
import requests

def create_user_admin(email: str, password: str, full_name: str, role: str = 'user') -> dict:
    """Creates a user via the Admin API and inserts profile row."""
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise Exception('SUPABASE_SERVICE_ROLE_KEY is required to create users programmatically')
    url = f"{SUPABASE_URL}/auth/v1/admin/users"
    headers = {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {"email": email, "password": password, "email_confirm": True}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code not in (200, 201):
        return {'error': r.text}
    user = r.json()
    # insert into profiles table
    profile = {
        'id': user['id'],
        'email': email,
        'full_name': full_name,
        'role': role
    }
    supabase.table('profiles').upsert(profile).execute()
    return user

# Authentication: we'll use Supabase's client-side auth for sign in

def sign_in(email: str, password: str):
    auth = supabase.auth.sign_in(email=email, password=password)
    return auth

def sign_out():
    supabase.auth.sign_out()

# ---------------------- Task CRUD ----------------------

def create_task(title: str, owner_id: str, description: str = '', due_date: Optional[date] = None, metadata: dict = None):
    payload = {
        'title': title,
        'description': description,
        'owner': owner_id,
        'status': 'todo',
        'due_date': due_date.isoformat() if due_date else None,
        'metadata': json.dumps(metadata or {})
    }
    res = supabase.table('tasks').insert(payload).execute()
    return res.data


def get_tasks_for_admin():
    res = supabase.table('tasks').select('*, profiles(*)').order('due_date', desc=False).execute()
    return res.data or []


def get_tasks_for_user(user_id: str):
    res = supabase.table('tasks').select('*').eq('owner', user_id).order('due_date', desc=False).execute()
    return res.data or []


def update_task(task_id: str, updates: dict):
    res = supabase.table('tasks').update(updates).eq('id', task_id).execute()
    return res.data


def delete_task(task_id: str):
    res = supabase.table('tasks').delete().eq('id', task_id).execute()
    return res.data

# ---------------------- UI Helpers ----------------------

def status_color_html(status: str, due_date_str: Optional[str], done: bool):
    """Return HTML badge for status with color rules:
    - todo -> red
    - inprogress -> yellow
    - done -> green
    - if overdue and not done -> half red/half yellow
    due_date_str is ISO date
    """
    today = date.today()
    overdue = False
    if due_date_str:
        try:
            d = datetime.fromisoformat(due_date_str).date()
        except Exception:
            try:
                d = date.fromisoformat(due_date_str)
            except Exception:
                d = None
        if d and d < today and not done:
            overdue = True
    if overdue:
        # half red half yellow using linear-gradient
        html = f"<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:linear-gradient(90deg,#ff4d4d 50%,#ffd966 50%);color:#000;font-weight:600;'>OVERDUE</div>"
        return html
    if status == 'todo':
        return "<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:#ff4d4d;color:#fff;font-weight:600;'>TODO</div>"
    if status == 'inprogress':
        return "<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:#ffd966;color:#000;font-weight:600;'>IN PROGRESS</div>"
    if status == 'done':
        return "<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:#4caf50;color:#fff;font-weight:600;'>DONE</div>"
    return f"<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:#cccccc;color:#000;font-weight:600;'>{status}</div>"

# ---------------------- AI Helpers ----------------------

def summarize_tasks_openai(tasks: list) -> str:
    """Use OpenAI to summarize completed tasks. Requires OPENAI_API_KEY and openai package."""
    if not OPENAI_API_KEY or openai is None:
        # fallback simple summary
        done = [t for t in tasks if t.get('status') == 'done']
        s = f"""تم إنجاز {len(done)} مهمة. عناوين المهام:
{'\n'.join([f"- {t.get('title')}" for t in done[:10]])}"""
s = f"""تم إنجاز {len(done)} مهمة. عناوين المهام:
{'\n'.join([f"- {t.get('title')}" for t in done[:10]])}"""
        return s
    texts = '\n'.join([f"{t.get('title')}: {t.get('description','')}" for t in tasks if t.get('status') == 'done'])
    prompt = "لخص التالي باختصار عربي نقاط مرتبة:\n" + texts
    try:
        resp = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=300,
            temperature=0.2
        )
        return resp.choices[0].text.strip()
    except Exception as e:
        return f"AI summary failed: {e}"


def analyze_performance(tasks: list) -> dict:
    """Basic performance analysis and simple predictive model.
    - average completion time
    - percent overdue
    - risk prediction using simple features (if sklearn available)
    """
    df = pd.DataFrame(tasks)
    result = {}
    if df.empty:
        return {'message': 'لا توجد مهام.'}
    # normalize dates
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
    df['completed_at'] = pd.to_datetime(df['completed_at'], errors='coerce')
    df['time_to_complete_days'] = (df['completed_at'] - df['created_at']).dt.total_seconds() / (3600*24)
    avg_complete = df['time_to_complete_days'].dropna().mean()
    overdue_count = df[(df['due_date'] < pd.Timestamp(date.today())) & (df['status'] != 'done')].shape[0]
    total = df.shape[0]
    result['avg_completion_days'] = float(avg_complete) if pd.notna(avg_complete) else None
    result['overdue_percent'] = (overdue_count / total) * 100
    # simple risk model: label overdue (1) vs not (0) for past tasks
    if LogisticRegression and len(df.dropna(subset=['time_to_complete_days'])) > 10:
        # build features: time_to_complete_days (for done tasks), days_until_due at creation
        df['days_until_due_at_create'] = (df['due_date'] - df['created_at']).dt.days
        df['label_overdue'] = ((df['due_date'] < df['completed_at']) | ((df['due_date'] < pd.Timestamp(date.today())) & (df['status'] != 'done'))).astype(int)
        feature_df = df[['time_to_complete_days', 'days_until_due_at_create']].dropna()
        labels = df.loc[feature_df.index, 'label_overdue']
        if len(feature_df) >= 10 and len(labels.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(feature_df, labels, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            result['risk_model_accuracy'] = float(score)
            # predict risk for open tasks
            open_tasks = df[df['status'] != 'done']
            if not open_tasks.empty:
                feats = open_tasks[['time_to_complete_days', 'days_until_due_at_create']].fillna(0)
                probs = model.predict_proba(feats)
                result['open_tasks_risk'] = dict(zip(open_tasks['id'].astype(str), probs[:,1].tolist()))
    return result

# ---------------------- Streamlit App ----------------------

st.set_page_config(page_title='AI Tasks Dashboard', layout='wide')

st.title('داشبورد متابعة مهام AI')

# Auth: simple email/password sign-in
if 'user' not in st.session_state:
    st.session_state.user = None

with st.sidebar:
    st.header('تسجيل الدخول')
    if not st.session_state.user:
        email = st.text_input('البريد الإلكتروني')
        password = st.text_input('كلمة المرور', type='password')
        if st.button('تسجيل دخول'):
            auth = sign_in(email, password)
            if auth and auth.get('user'):
                st.success('تم تسجيل الدخول')
                st.session_state.user = auth['user']
            else:
                st.error('فشل تسجيل الدخول. تأكد من البريد/كلمة المرور.')
    else:
        st.markdown(f"مستخدم حالي: **{st.session_state.user.get('email')}**")
        if st.button('تسجيل خروج'):
            sign_out()
            st.session_state.user = None
            st.experimental_rerun()

# Require login
if not st.session_state.user:
    st.info('الرجاء تسجيل الدخول للمتابعة. يمكن للمسؤول إنشاء حسابات عبر لوحة المدير.')
    st.stop()

# get profile and role
current_user = get_profile_by_id(st.session_state.user['id'])
if not current_user:
    st.error('ملف المستخدم غير موجود في جدول profiles. تأكد من إنشاء profile.')
    st.stop()
role = current_user.get('role', 'user')

st.sidebar.markdown(f"**دورك:** {role}")

# Navigation
if role == 'admin':
    page = st.sidebar.selectbox('القائمة', ['Admin Dashboard', 'إضافة مستخدم', 'كل المهام'])
else:
    page = st.sidebar.selectbox('القائمة', ['مهامي', 'التحليل والتقارير'])

# ---------------------- Admin Pages ----------------------
if role == 'admin' and page == 'إضافة مستخدم':
    st.header('إنشاء مستخدم جديد')
    with st.form('create_user'):
        new_email = st.text_input('البريد الإلكتروني')
        new_password = st.text_input('كلمة المرور (مؤقتة)', type='password')
        new_fullname = st.text_input('الاسم الكامل')
        new_role = st.selectbox('دور المستخدم', ['user','admin'])
        submitted = st.form_submit_button('إنشاء')
        if submitted:
            try:
                res = create_user_admin(new_email, new_password, new_fullname, new_role)
                if res.get('error'):
                    st.error(res['error'])
                else:
                    st.success('تم إنشاء المستخدم بنجاح')
            except Exception as e:
                st.error(f'فشل إنشاء المستخدم: {e}')

if role == 'admin' and page == 'كل المهام':
    st.header('كل المهام (Admin)')
    tasks = get_tasks_for_admin()
    df = pd.DataFrame(tasks)
    if df.empty:
        st.info('لا توجد مهام')
    else:
        # Show interactive table
        st.dataframe(df[['id','title','owner','status','due_date']])
        st.markdown('---')
        # CRUD
        st.subheader('إنشاء مهمة جديدة')
        with st.form('create_task_admin'):
            title = st.text_input('عنوان المهمة')
            desc = st.text_area('وصف')
            owner_select = st.selectbox('اختر المالك', options=[(u['id'], u.get('email')) for u in list_users()], format_func=lambda x: x[1])
            due = st.date_input('تاريخ الإنجاز المتوقع', value=date.today())
            create_btn = st.form_submit_button('إنشاء')
            if create_btn:
                create_task(title, owner_select[0], desc, due, metadata={})
                st.success('تم إنشاء المهمة')
                st.experimental_rerun()
        st.markdown('---')
        st.subheader('تعديل / حذف مهمة')
        task_ids = df['id'].tolist()
        sel = st.selectbox('اختر مهمة', options=task_ids)
        if sel:
            t = df[df['id'] == sel].iloc[0].to_dict()
            st.write('العنوان:', t['title'])
            new_status = st.selectbox('الحالة', ['todo','inprogress','done'], index=['todo','inprogress','done'].index(t['status']))
            if st.button('تحديث الحالة'):
                updates = {'status': new_status}
                if new_status == 'done':
                    updates['completed_at'] = datetime.utcnow().isoformat()
                update_task(sel, updates)
                st.success('تم التحديث')
                st.experimental_rerun()
            if st.button('حذف المهمة'):
                delete_task(sel)
                st.success('تم الحذف')
                st.experimental_rerun()

if role == 'admin' and page == 'Admin Dashboard':
    st.header('لوحة المدير')
    # Overview KPIs
    tasks = get_tasks_for_admin()
    df = pd.DataFrame(tasks)
    total = len(df)
    overdue = 0
    inprogress = 0
    todo = 0
    done = 0
    if not df.empty:
        df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
        due_today = df[df['due_date'] < pd.Timestamp(date.today())]
        overdue = due_today[due_today['status'] != 'done'].shape[0]
        todo = df[df['status']=='todo'].shape[0]
        inprogress = df[df['status']=='inprogress'].shape[0]
        done = df[df['status']=='done'].shape[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('مجموع المهام', total)
    col2.metric('متأخرة', overdue)
    col3.metric('قيد العمل', inprogress)
    col4.metric('منجزة', done)

    st.markdown('### توزيع الحالات')
    if not df.empty:
        fig = px.pie(df, names='status', title='حالة المهام')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('### المهام حسب المستخدم')
    if not df.empty:
        merged = []
        for t in tasks:
            p = get_profile_by_id(t.get('owner'))
            merged.append({'title':t.get('title'), 'owner': p.get('email') if p else '—', 'status':t.get('status'), 'due_date':t.get('due_date')})
        mdf = pd.DataFrame(merged)
        fig2 = px.histogram(mdf, x='owner', color='status', barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('---')
    st.subheader('تنبيهات سريعة')
    # Alerts
    if not df.empty:
        alerts = []
        for t in tasks:
            due = t.get('due_date')
            st_html = status_color_html(t.get('status'), due, t.get('status')=='done')
            if t.get('status') != 'done' and t.get('due_date') and pd.to_datetime(t.get('due_date')).date() < date.today():
                alerts.append(f"المهمة '{t.get('title')}' ل{get_profile_by_id(t.get('owner')).get('email', '—')} متأخرة")
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success('لا توجد تنبيهات حالية')

    st.markdown('---')
    st.subheader('تحليل الأداء (AI)')
    if st.button('تشغيل تحليل الأداء'):        
        perf = analyze_performance(tasks)
        st.json(perf)
    if st.button('تلخيص المهام المنجزة'):
        summ = summarize_tasks_openai(tasks)
        st.markdown('**ملخص:**')
        st.write(summ)

# ---------------------- User Pages ----------------------
if role == 'user':
    if page == 'مهامي':
        st.header('مهامي')
        tasks = get_tasks_for_user(current_user['id'])
        if not tasks:
            st.info('لا توجد مهام مخصصة لك.')
        else:
            for t in tasks:
                cols = st.columns([6,2,2,1])
                with cols[0]:
                    st.markdown(f"### {t.get('title')}")
                    st.write(t.get('description'))
                    st.write('تاريخ الاستحقاق:', t.get('due_date'))
                with cols[1]:
                    st.markdown(status_color_html(t.get('status'), t.get('due_date'), t.get('status')=='done'), unsafe_allow_html=True)
                with cols[2]:
                    new_status = st.selectbox(f"تغيير الحالة {t.get('id')}", ['todo','inprogress','done'], index=['todo','inprogress','done'].index(t.get('status')))
                    if st.button(f'تحديث {t.get("id")}', key=f'up_{t.get("id")}'):
                        updates = {'status': new_status}
                        if new_status == 'done':
                            updates['completed_at'] = datetime.utcnow().isoformat()
                        update_task(t.get('id'), updates)
                        st.success('تم التحديث')
                        st.experimental_rerun()
                with cols[3]:
                    if st.button(f'تفاصيل {t.get("id")}', key=f'det_{t.get("id")}'):
                        st.json(t)

    if page == 'التحليل والتقارير':
        st.header('تحليل وتقارير شخصية')
        tasks = get_tasks_for_user(current_user['id'])
        if not tasks:
            st.info('لا توجد بيانات للتحليل')
        else:
            df = pd.DataFrame(tasks)
            st.subheader('مخطط الحالة')
            fig = px.histogram(df, x='status')
            st.plotly_chart(fig, use_container_width=True)
            st.subheader('تحليل أداء')
            perf = analyze_performance(tasks)
            st.json(perf)
            st.subheader('تلخيص المهام المنجزة')
            summ = summarize_tasks_openai(tasks)
            st.write(summ)

# --------------- End of App ---------------

st.markdown('---')
st.caption('تم الإنشاء بواسطة نظام مساعدة. يمكن تخصيص الكود وإضافة ميزات إضافية عند الطلب.')
