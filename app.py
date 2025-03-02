from flask import Flask, render_template, request, jsonify, send_file, redirect
from flask_migrate import Migrate
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tradingview_screener import Query, col
from datetime import datetime, timedelta
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
import time
import csv
import io
import numpy as np
import os
import pandas as pd

load_dotenv()

# --------------------------
# App Configuration
# --------------------------
app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']

if os.environ.get('RENDER') and not app.secret_key:
    raise ValueError("SECRET_KEY must be set in production environment")

CORS(app, supports_credentials=True)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL'].replace('postgres://', 'postgresql://')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {  # Connection pooling
    'pool_size': 10,
    'max_overflow': 30,
    'pool_recycle': 300,
    'pool_pre_ping': True
}

db = SQLAlchemy(app)
migrate = Migrate(app, db)

app.config['SECURITY_PASSWORD_SALT'] = os.environ.get('SECURITY_PASSWORD_SALT', 'another-secret-salt-123')
serializer = URLSafeTimedSerializer(app.secret_key)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

# --------------------------
# Database Models
# --------------------------
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(200), nullable=False)
    security_question1 = db.Column(db.String(200))
    security_answer1 = db.Column(db.String(255), nullable=False)
    security_question2 = db.Column(db.String(200))
    security_answer2 = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = UserSettings(
            exchanges=['BYBIT'],
            types=['swap'],
            market='crypto'
        )

    def set_security_answers(self, answer1, answer2):
        self.security_answer1 = generate_password_hash(answer1)
        self.security_answer2 = generate_password_hash(answer2)

    def check_security_answers(self, answer1, answer2):
        return check_password_hash(self.security_answer1, answer1) and check_password_hash(self.security_answer2, answer2)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class UserSettings(db.Model):
    __tablename__ = 'user_settings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        unique=True)
    exchanges = db.Column(db.JSON, default=['BYBIT'])
    types = db.Column(db.JSON, default=['swap'])
    market = db.Column(db.String(20), default='crypto')

    user = db.relationship('User', backref=db.backref('settings', uselist=False, cascade='all, delete-orphan'))

class WatchlistItem(db.Model):
    __tablename__ = 'watchlist_items'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    date_added = db.Column(db.String(20), nullable=False)
    time_added = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    rsi_4hr = db.Column(db.Float, nullable=False)
    rsi_daily = db.Column(db.Float, nullable=False)
    ema_4hr = db.Column(db.String(20), nullable=False)
    ema_daily = db.Column(db.String(20), nullable=False)
    grade = db.Column(db.String(1), nullable=False)

# --------------------------
# Database Initialization
# --------------------------
with app.app_context():
    db.create_all()
    try:
        if not User.query.filter_by(is_admin=True).first():
            admin = User(
                username=os.environ.get('ADMIN_USERNAME', 'admin'),
                email=os.environ.get('ADMIN_EMAIL', 'admin@example.com'),
                security_question1='what?',
                security_answer1=generate_password_hash(os.environ.get('ADMIN_ANSWER1', 'default')),
                security_question2='who?',
                security_answer2=generate_password_hash(os.environ.get('ADMIN_ANSWER2', 'default')),
                is_admin=True
            )
            admin.set_password(os.environ.get('ADMIN_PASSWORD', 'admin'))

            db.session.add(admin)
            db.session.commit()

            admin_settings = UserSettings(user_id=admin.id)
            db.session.add(admin_settings)
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Admin initialization: {str(e)}")

# --------------------------
# Authentication Setup
# --------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized_callback():
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Unauthorized'}), 401
    return redirect(f'/login?next={request.path}')

# --------------------------
# Crypto Screener Class
# --------------------------
class CryptoScreener:
    def __init__(self, user_settings=None):
        self.user_settings = user_settings or {
            'exchanges': ['BYBIT'],
            'types': ['swap'],
            'market': 'crypto'
        }
        self.data_frame = None

    def fetch_data(self):
        try:
            query = (Query()
                     .limit(1000)
                     .select(
                'name', 'close', 'close|240',
                'RSI|240', 'RSI', 'RSI|1W',
                'EMA20|240', 'EMA50|240', 'EMA100|240',
                'EMA20', 'EMA50', 'EMA100',
                'EMA20|1W', 'EMA50|1W', 'EMA100|1W',
                'volume', 'Value.Traded',
                'Low.3M', 'High.3M',
                'High.6M', 'Low.6M',
                'price_52_week_high', 'price_52_week_low',
                'High.All', 'Low.All'
            )
                     .where(
                col('exchange').isin(self.user_settings['exchanges']),
                col('type').isin(self.user_settings['types']),
                col('typespecs').has('perpetual'),
                col('currency').like('USDT')
            )
                     .set_markets(self.user_settings['market'])
                     .get_scanner_data()
                     )
            self.data_frame = query[1]
            self._process_data()
            return self.data_frame
        except Exception as e:
            print(f"Data error: {str(e)}")
            return None

    def _process_data(self):
        self.data_frame[['EMA100|240_class', 'EMA100_class']] = self.data_frame.apply(
            self.classify_ema, axis=1, result_type="expand"
        )

    def classify_ema(self, row):
        def classify(timeframe):
            close = row.get(f'close{timeframe}', float('nan'))
            EMA20 = row.get(f'EMA20{timeframe}', float('nan'))
            EMA50 = row.get(f'EMA50{timeframe}', float('nan'))
            EMA100 = row.get(f'EMA100{timeframe}', float('nan'))

            if pd.isna(close) or pd.isna(EMA20) or pd.isna(EMA50) or pd.isna(EMA100):
                return "N/A"
            if close > EMA20 > EMA50 > EMA100:
                return "AOTS+"
            elif close < EMA20 < EMA50 < EMA100:
                return "iAOTS+"
            elif close > EMA100:
                return "ZS"
            elif close < EMA100:
                return "iZS"
            elif EMA20 > EMA50 > EMA100:
                return "AOTS"
            elif EMA20 < EMA50 < EMA100:
                return "iAOTS"
            else:
                return "Spaghetti"

        ema_status_240 = classify('|240')
        ema_status_daily = classify('')
        return pd.Series([ema_status_240, ema_status_daily])

    def apply_alpha_filter(self, ema_4hr=None, rsi_4hr=None, ema_daily=None, rsi_daily=None):
        if self.data_frame is None:
            self.fetch_data()

        filtered_data = self.data_frame.copy()

        if ema_4hr and ema_4hr != "None":
            ema_col = f"EMA{ema_4hr}|240"
            filtered_data = filtered_data[filtered_data['close|240'] > filtered_data[ema_col]]
        if rsi_4hr and rsi_4hr != "None":
            filtered_data = filtered_data[filtered_data['RSI|240'] > float(rsi_4hr)]

        # Daily Filters
        if ema_daily and ema_daily != "None":
            ema_col = f"EMA{ema_daily}"
            filtered_data = filtered_data[filtered_data['close'] > filtered_data[ema_col]]
        if rsi_daily and rsi_daily != "None":
            filtered_data = filtered_data[filtered_data['RSI'] > float(rsi_daily)]

        return filtered_data

crypto_screener = CryptoScreener()

# --------------------------
# Template Routes
# --------------------------
@app.route('/')
def index():
    default_exchange = 'BYBIT'
    if current_user.is_authenticated and current_user.settings:
        exchange = current_user.settings.exchanges[0] if current_user.settings.exchanges else default_exchange
    else:
        exchange = default_exchange
    return render_template('index.html', current_exchange=exchange)

@app.route('/admin')
@login_required
def admin_page():
    if not current_user.is_admin:
        return redirect('/')
    return render_template('admin.html')

@app.route('/settings')
@login_required
def settings_page():
    return render_template('settings.html')

@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect('/')
    return render_template('login.html')

@app.route('/watchlist_page')
@login_required
def watchlist_page():
    return render_template('watchlist.html')

# --------------------------
# Authentication Routes
# --------------------------
@app.route('/check_login')
def check_login():
    if current_user.is_authenticated:
        return jsonify({
            'logged_in': True,
            'username': current_user.username,
            'is_admin': current_user.is_admin
        })
    return jsonify({'logged_in': False})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()

    if not user or not user.check_password(data.get('password', '')):
        return jsonify({
            'success': False,
            'message': 'Invalid username or password'
        }), 401

    login_user(user)
    return jsonify({'success': True, 'message': 'Login successful'})

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True})

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        required = ['username', 'email', 'password',
                    'security_question1', 'security_answer1',
                    'security_question2', 'security_answer2']
        if not all(field in data for field in required):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400

        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already exists'}), 400

        user = User(
            username=data['username'],
            email=data['email'],
            security_question1=data['security_question1'],
            security_question2=data['security_question2']
        )
        user.set_password(data['password'])
        user.set_security_answers(
            data['security_answer1'],
            data['security_answer2']
        )

        db.session.add(user)
        db.session.commit()
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get_security_questions', methods=['POST'])
def get_security_questions():
    data = request.json
    user = User.query.filter_by(
        email=data.get('email'),
        username=data.get('username')
    ).first()

    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    return jsonify({
        'success': True,
        'question1': user.security_question1,
        'question2': user.security_question2
    })

@app.route('/reset_password_request', methods=['POST'])
def reset_password_request():
    try:
        data = request.json
        user = User.query.filter_by(
            email=data['email'],
            username=data['username']
        ).first()

        if not user:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

        if not user.check_security_answers(data['answer1'], data['answer2']):
            return jsonify({'success': False, 'message': 'Invalid security answers'}), 401

        exp_timestamp = time.mktime((datetime.utcnow() + timedelta(minutes=10)).timetuple())

        reset_token = serializer.dumps({
            'user_id': user.id,
            'exp': exp_timestamp
        }, salt='password-reset')

        return jsonify({'success': True, 'token': reset_token})

    except Exception as e:
        app.logger.error(f"Password reset error: {str(e)}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

# @app.route('/verify_reset_code', methods=['POST'])
# def verify_reset_code():
#     data = request.json
#     reset_data = session.get('reset_data')
#
#     if not reset_data:
#         return jsonify({'success': False, 'message': 'Invalid session'}), 400
#
#     if datetime.utcnow() > reset_data['expires']:
#         return jsonify({'success': False, 'message': 'Code expired'}), 400
#
#     if data['code'] != reset_data['code']:
#         return jsonify({'success': False, 'message': 'Invalid code'}), 400
#
#     return jsonify({'success': True, 'token': reset_data['token']})

@app.route('/reset_password', methods=['POST'])
def reset_password():
    try:
        data = request.json
        token_data = serializer.loads(
            data['token'],
            salt='password-reset',
            max_age=600
        )

        user = User.query.get(token_data['user_id'])
        if not user:
            return jsonify({'success': False, 'message': 'Invalid token'}), 400

        if data['new_password'] != data['confirm_password']:
            return jsonify({'success': False, 'message': 'Passwords mismatch'}), 400

        user.set_password(data['new_password'])
        db.session.commit()

        return jsonify({'success': True, 'message': 'Password updated successfully'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

# --------------------------
# Data Routes
# --------------------------
@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    def serialize_data(df):
        return df.replace({np.nan: None}).to_dict(orient='records')

    try:
        if current_user.is_authenticated:
            if not current_user.settings:
                current_user.settings = UserSettings()
                db.session.commit()

            settings = {
                'exchanges': current_user.settings.exchanges or ['BYBIT'],
                'types': current_user.settings.types or ['swap'],
                'market': current_user.settings.market or 'crypto'
            }
        else:
            settings = {
                'exchanges': ['BYBIT'],
                'types': ['swap'],
                'market': 'crypto'
            }

        screener = CryptoScreener(settings)
        data = screener.fetch_data()
        return jsonify(serialize_data(data)) if data is not None else jsonify([])
    except Exception as e:
        print(f"Data error: {str(e)}")
        return jsonify([]), 500

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    def serialize_data(df):
        return df.replace({np.nan: None}).to_dict(orient='records')

    try:
        if current_user.is_authenticated:
            settings = {
                'exchanges': getattr(current_user.settings, 'exchanges', ['BYBIT']),
                'types': getattr(current_user.settings, 'types', ['swap']),
                'market': getattr(current_user.settings, 'market', 'crypto')
            }
        else:
            settings = {
                'exchanges': ['BYBIT'],
                'types': ['swap'],
                'market': 'crypto'
            }

        screener = CryptoScreener(settings)
        filters = request.json
        filtered_data = screener.apply_alpha_filter(
            ema_4hr=filters.get('ema_4hr'),
            rsi_4hr=filters.get('rsi_4hr'),
            ema_daily=filters.get('ema_daily'),
            rsi_daily=filters.get('rsi_daily')
        )
        return jsonify(serialize_data(filtered_data))
    except Exception as e:
        print(f"Filtering error: {str(e)}")
        return jsonify([]), 500

# --------------------------
# Watchlist Routes
# --------------------------
@app.route('/watchlist', methods=['GET', 'POST', 'DELETE'])
@login_required
def manage_watchlist():
    if request.method == 'GET':
        items = WatchlistItem.query.filter_by(user_id=current_user.id).all()
        return jsonify([{
            'id': item.id,
            'date_added': item.date_added,
            'time_added': item.time_added,
            'name': item.name,
            'price': item.price,
            'volume': item.volume,
            'rsi_4hr': item.rsi_4hr,
            'rsi_daily': item.rsi_daily,
            'ema_4hr': item.ema_4hr,
            'ema_daily': item.ema_daily,
            'grade': item.grade
        } for item in items])

    if request.method == 'POST':
        data = request.json
        now = datetime.now()
        new_item = WatchlistItem(
            user_id=current_user.id,
            date_added=now.strftime("%Y-%m-%d"),
            time_added=now.strftime("%H:%M:%S"),
            name=data['name'],
            price=data['price'],
            volume=data['volume'],
            rsi_4hr=data['rsi_4hr'],
            rsi_daily=data['rsi_daily'],
            ema_4hr=data['ema_4hr'],
            ema_daily=data['ema_daily'],
            grade=data['grade']
        )
        db.session.add(new_item)
        db.session.commit()
        return jsonify({'success': True})

    if request.method == 'DELETE':
        item = WatchlistItem.query.get(request.json['id'])
        if item and item.user_id == current_user.id:
            db.session.delete(item)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'error': 'Item not found'}), 404

@app.route('/watchlist/<int:item_id>', methods=['DELETE'])
@login_required
def delete_watchlist_item(item_id):
    item = WatchlistItem.query.get(item_id)
    if item and item.user_id == current_user.id:
        db.session.delete(item)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'Item not found or unauthorized'}), 404

@app.route('/export_watchlist', methods=['GET'])
@login_required
def export_watchlist():
    try:
        # Get watchlist items from database
        items = WatchlistItem.query.filter_by(user_id=current_user.id).all()

        # Create CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date Added", "Time Added", "Name", "Price", "Volume",
                         "RSI 4hr", "RSI Daily", "EMA 4hr", "EMA Daily", "Grade"])

        for item in items:
            writer.writerow([
                item.date_added,
                item.time_added,
                item.name,
                item.price,
                item.volume,
                item.rsi_4hr,
                item.rsi_daily,
                item.ema_4hr,
                item.ema_daily,
                item.grade
            ])

        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='watchlist.csv'
        )
    except Exception as e:
        print(f"Error exporting watchlist: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_watchlist', methods=['POST'])
@login_required
def clear_watchlist():
    try:
        # Delete all watchlist items for the current user from the database
        WatchlistItem.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({"status": "success", "message": "Watchlist cleared"})
    except Exception as e:
        db.session.rollback()
        print(f"Error clearing watchlist: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --------------------------
# Settings Routes
# --------------------------
@app.route('/api/settings', methods=['GET', 'PUT'])
@login_required
def handle_settings():
    if not current_user.settings:
        current_user.settings = UserSettings(
            exchanges=['BYBIT'],
            types=['swap'],
            market='crypto'
        )
        db.session.commit()

    if request.method == 'GET':
        return jsonify({
            'exchanges': current_user.settings.exchanges,
            'types': current_user.settings.types,
            'market': current_user.settings.market
        })

    if request.method == 'PUT':
        data = request.json
        try:
            current_user.settings.exchanges = data.get('exchanges', ['BYBIT'])
            current_user.settings.types = data.get('types', ['swap'])
            current_user.settings.market = data.get('market', 'crypto')
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

# --------------------------
# Admin Routes
# --------------------------
@app.route('/admin/users', methods=['GET'])
@login_required
def get_users():
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    users = User.query.all()
    return jsonify([{
        'id': user.id,
        'username': user.username,
        'is_admin': user.is_admin,
        'email': user.email,
        'security_question1': user.security_question1,
        'security_answer1': user.security_answer1,
        'security_question2': user.security_question2,
        'security_answer2': user.security_answer2
    } for user in users])

@app.route('/admin/user/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"error": "User not found"}), 404

@app.route('/admin/reset-password/<int:user_id>', methods=['POST'])
@login_required
def admin_reset_password(user_id):
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    user = User.query.get(user_id)
    if user:
        user.set_password("defaultpassword")
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"error": "User not found"}), 404

# --------------------------
# Main Entry Point
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)