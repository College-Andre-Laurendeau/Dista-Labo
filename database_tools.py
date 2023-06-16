#TODO Replace all this with a proper object model (consider using sqlalchemy)
import os, sys
from config import *

dbp = ()

def connect_database(user, args):
    global dbp
    
    creation_scripts = []
    args = parse_args()
    
    if not args.db :
        import sqlite3
        dbp = ('?',)
        database = sqlite3.connect(':memory:')
        creation_scripts = [user + '/Documents/dista/logisticam/db/schema_sqlite.sql',
            user + '/Documents/dista/logisticam/db/statistics.sql']
    
    elif args.db.startswith('mysql:') :
        import mysql.connector
        dbp = ('%s',)
        dbpath = args.db[args.db.index(':') + 1 : ].rsplit('/', 1)
        database = mysql.connector.connect(
            host=dbpath[-2] if len(dbpath) > 1 else 'localhost',
            user='dista',
            password='dista-sql',
            database=dbpath[-1] if dbpath[-1] else 'dista')
        creation_scripts = [user + '/Documents/dista/logisticam/db/set_timezone.sql']

    elif args.db.startswith('mariadb:') :
        import mariadb
        dbp = ('?',)
        dbpath = args.db[args.db.index(':') + 1 : ].rsplit('/', 1)
        database = mariadb.connect(
            host=dbpath[-2] if len(dbpath) > 1 else 'localhost',
            user='gripcal',
            password='dista-SQL-write',
            database=dbpath[-1] if dbpath[-1] else 'dista-dev')
        creation_scripts = [user + '/Documents/dista/logisticam/db/set_timezone.sql']

    else :
        import sqlite3
        dbp = ('?',)
        dbname = args.db[args.db.find(':') + 1 : ] # assuming we might have 'sqlite:...'
        dbpath = user + '/Documents/dista/' + (dbname or 'dista.sqlite')
        if not os.path.isfile(dbpath) :
            creation_scripts = [user + '/Documents/dista/logisticam/db/schema_sqlite.sql',
                user + '/Documents/dista/logisticam/db/statistics.sql']
        database = sqlite3.connect(dbpath)
    
    for script in creation_scripts :
        with open(script, 'r') as f :
            database.executescript(f.read())

    return database


def load_cameras(args, database):
    cur = database.cursor()
    
    serial_numbers = []
    ip_addresses = []
    cam_ports = []

    for camera in args.cameras :
        address, sn = camera.split('/') if '/' in camera else ('', camera)
        machine_id = None
        port_no = '5555'

        if address :
            # Check if ip (or hostname) exists in database
            ip, port_no = address.split(':') if ':' in address else (address, port_no)
            cur.execute(
                "SELECT id FROM Machines WHERE address=%s ORDER BY id DESC" % dbp, (ip,))
            machine_id = cur.fetchone()
            if machine_id :
                machine_id = machine_id[0]
            else :
                cur.execute("INSERT INTO Machines (address) VALUES (%s)" % dbp, (ip,))
                machine_id = cur.lastrowid
            ip_addresses.append(ip)
            cam_ports.append(int(port_no))

        # Check if serial exists in database
        cur.execute(
            "SELECT machine FROM Cameras WHERE serial=%s ORDER BY id DESC" % dbp, (sn,))
        cam = cur.fetchone()
        if not cam or cam[0] != machine_id :
            if machine_id :
                cur.execute(
                    """INSERT INTO Cameras (serial, resolution, machine, port_no)
                    VALUES (%s, %s, %s, %s)""" % (dbp*4),
                    (sn, args.resolution, machine_id, port_no))
            else :
                cur.execute(
                    "INSERT INTO Cameras (serial, resolution) VALUES (%s, %s)" % (dbp*2),
                    (sn, args.resolution))
        serial_numbers.append(sn)
    
    if not serial_numbers :
        # Load all cameras and machines from database (ie. restore last configuration)
        #TODO Consider reloading all args from last session command_args instead
        cur.execute("SELECT cam_serial, ip_address FROM Camera_configurations")
        for row in cur.fetchall() :
            serial_numbers.append(int(row[0]))
            if row[1] :
                address = row[1].split(':')
                ip_addresses.append(address[0])
                cam_ports.append(int(address[1]))

        # Manual overrides (when debugging only!)
        # serial_numbers = [22548184, 20716499, 23336181, 27645295, 24929741]
        # ip_addresses = ['192.168.2.54', '192.168.2.45', '192.168.1.100']
        # cam_ports = [5555, 5560, 5565]

        if not serial_numbers :
            sys.exit("FATAL: No cameras defined")
    
    # Record the start of a new session
    cur.execute("INSERT INTO Sessions (command_args) VALUES (%s)" % dbp,
        (' '.join(sys.argv),))
    
    database.commit()
    cur.close()
    
    return serial_numbers, ip_addresses, cam_ports
    #TODO Return an array of camera objects instead


def store_frame(database, serial_number):
    #sql = "INSERT INTO Frames (created_by) VALUES (%s)" % dbp #FIXME this is the right logic
    sql = "INSERT INTO Frames (created_by) SELECT id FROM Cameras WHERE serial=%s" % dbp
    cur = database.cursor()
    cur.execute(sql, (serial_number,))
    frame_id = cur.lastrowid
    database.commit()
    cur.close()
    return frame_id


def store_detections(database, frame_id, cap):
    cur = database.cursor()
    
    if len(cap.detecs) == 0 :
        return

    for i in range(len(cap.object_center_positions)) :
        x = int(100 * cap.object_center_positions[i][0])
        y = int(100 * cap.object_center_positions[i][1])
        z = int(100 * cap.object_center_positions[i][2])
        sql = "INSERT INTO Detections VALUES (%s, %s, %s, %s, %s)" % (dbp*5)
        cur.execute(sql, (frame_id, i, x, z, y))
        #FIXME we shouldn't invert y and z when we implement global coordinates,
        # but doing so currently makes more sense with relative coordinates.

    if len(cap.object_center_positions) > 1 :
        for i in range(len(cap.object_center_positions)) :
            for j in range(i+1, len(cap.object_center_positions)) :
                dist = 100 * cap.distmat[i,j]
                sql = "INSERT INTO Distances VALUES (%s, %s, %s, %s)" % (dbp*4)
                cur.execute(sql, (frame_id, i, j, float(dist)))

    database.commit()
    cur.close()
