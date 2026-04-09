from flask import Blueprint, render_template, request, session, redirect, url_for, flash
import sqlite3
import os
from functools import wraps
from werkzeug.security import check_password_hash

admin_bp = Blueprint('admin', __name__, template_folder='templates/admin')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'users.db')

ITEMS_PER_PAGE = 15

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def admin_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Please log in as admin.', 'error')
            return redirect(url_for('admin.admin_login'))
        return f(*args, **kwargs)
    return decorated

# ─── LOGIN ────────────────────────────────
@admin_bp.route('/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin_id' in session:
        return redirect(url_for('admin.admin_dashboard'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        with get_db() as conn:
            admin = conn.execute("SELECT * FROM admins WHERE email = ?", (email,)).fetchone()
        if admin and check_password_hash(admin['password'], password):
            session['admin_id']   = admin['id']
            session['admin_name'] = admin['name']
            return redirect(url_for('admin.admin_dashboard'))
        flash('Invalid admin credentials.', 'error')
    return render_template('admin/admin_login.html')

# ─── LOGOUT ───────────────────────────────
@admin_bp.route('/logout')
def admin_logout():
    session.pop('admin_id',   None)
    session.pop('admin_name', None)
    return redirect(url_for('admin.admin_login'))

# ─── DASHBOARD ────────────────────────────
@admin_bp.route('/dashboard')
@admin_login_required
def admin_dashboard():
    with get_db() as conn:
        total_users   = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        total_emails  = conn.execute("SELECT COUNT(*) FROM email_logs").fetchone()[0]
        emails_ok     = conn.execute("SELECT COUNT(*) FROM email_logs WHERE status='sent'").fetchone()[0]
        emails_fail   = total_emails - emails_ok

        # Disease breakdown for chart
        disease_rows = conn.execute(
            "SELECT disease, COUNT(*) as cnt FROM reports GROUP BY disease"
        ).fetchall()

        # Last 7 days activity
        daily_rows = conn.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as cnt
            FROM reports
            WHERE created_at >= DATE('now', '-6 days')
            GROUP BY day ORDER BY day
        """).fetchall()

        recent_reports = conn.execute(
            "SELECT * FROM reports ORDER BY created_at DESC LIMIT 5"
        ).fetchall()

    chart_labels  = [r['disease'] for r in disease_rows]
    chart_data    = [r['cnt']     for r in disease_rows]
    daily_labels  = [r['day'] for r in daily_rows]
    daily_data    = [r['cnt'] for r in daily_rows]

    return render_template('admin/admin_dashboard.html',
        total_users=total_users,
        total_reports=total_reports,
        total_emails=total_emails,
        emails_ok=emails_ok,
        emails_fail=emails_fail,
        chart_labels=chart_labels,
        chart_data=chart_data,
        daily_labels=daily_labels,
        daily_data=daily_data,
        recent_reports=recent_reports,
    )

# ─── USERS ────────────────────────────────
@admin_bp.route('/users')
@admin_login_required
def admin_users():
    search = request.args.get('q', '').strip()
    page   = max(1, int(request.args.get('page', 1)))
    offset = (page - 1) * ITEMS_PER_PAGE

    with get_db() as conn:
        if search:
            users = conn.execute(
                "SELECT id, name, email, created_at FROM users WHERE name LIKE ? OR email LIKE ? LIMIT ? OFFSET ?",
                (f'%{search}%', f'%{search}%', ITEMS_PER_PAGE, offset)
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM users WHERE name LIKE ? OR email LIKE ?",
                (f'%{search}%', f'%{search}%')
            ).fetchone()[0]
        else:
            users = conn.execute(
                "SELECT id, name, email, created_at FROM users LIMIT ? OFFSET ?",
                (ITEMS_PER_PAGE, offset)
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    total_pages = max(1, -(-total // ITEMS_PER_PAGE))
    return render_template('admin/admin_users.html',
        users=users, search=search, page=page,
        total_pages=total_pages, total=total)

# ─── REPORTS ──────────────────────────────
@admin_bp.route('/reports')
@admin_login_required
def admin_reports():
    disease_filter  = request.args.get('disease', '')
    severity_filter = request.args.get('severity', '')
    page   = max(1, int(request.args.get('page', 1)))
    offset = (page - 1) * ITEMS_PER_PAGE

    query  = "SELECT * FROM reports WHERE 1=1"
    params = []
    if disease_filter:
        query  += " AND disease = ?"
        params.append(disease_filter)
    if severity_filter:
        query  += " AND severity = ?"
        params.append(severity_filter)

    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"

    with get_db() as conn:
        total   = conn.execute(count_query, params).fetchone()[0]
        reports = conn.execute(query, params + [ITEMS_PER_PAGE, offset]).fetchall()

    total_pages = max(1, -(-total // ITEMS_PER_PAGE))
    return render_template('admin/admin_reports.html',
        reports=reports,
        disease_filter=disease_filter,
        severity_filter=severity_filter,
        page=page,
        total_pages=total_pages,
        total=total)

# ─── EMAILS ───────────────────────────────
@admin_bp.route('/emails')
@admin_login_required
def admin_emails():
    page   = max(1, int(request.args.get('page', 1)))
    offset = (page - 1) * ITEMS_PER_PAGE
    with get_db() as conn:
        total  = conn.execute("SELECT COUNT(*) FROM email_logs").fetchone()[0]
        emails = conn.execute(
            "SELECT * FROM email_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (ITEMS_PER_PAGE, offset)
        ).fetchall()
    total_pages = max(1, -(-total // ITEMS_PER_PAGE))
    return render_template('admin/admin_emails.html',
        emails=emails, page=page, total_pages=total_pages, total=total)
