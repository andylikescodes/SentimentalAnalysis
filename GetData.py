import sqlite3
import xlrd
import time
from DataProcessing import DataProcessor

class Database:
	def __init__(self, database):
		self.database = database
		self.conn = sqlite3.connect(self.database)
		self.c = self.conn.cursor()
		self.processor = DataProcessor()

	# retrieve data from test.db
	# inputs:
	# database - name of database
	# table - name of table in database
	# column - name of the column
	# whereColumn - the column for the where clause
	# whereValue - the value for the WHERE parameter in SQL
	def readDB(self, table, column, whereColumn = None, whereValue = None):
		if whereColumn != None and whereValue != None:
			sql = "SELECT %s FROM %s WHERE %s =" % (column, table, whereColumn) + str(whereValue) 
			return [eachTweet[0]for eachTweet in self.c.execute(sql)]
		else:
			sql = "SELECT %s From %s" % (column,table)
			return [eachTweet[0]for eachTweet in self.c.execute(sql)]

	# transform excelsheet with scores, two columns, into a sqlite3 database
	def transformExcelScore(self, excelFile): 
		workbook = xlrd.open_workbook(excelFile)
		worksheets = workbook.sheet_names()
		for worksheet_name in worksheets:
			worksheet = workbook.sheet_by_name(worksheet_name)
			sql='CREATE TABLE %s (Tweet TEXT, Score REAL)' % (worksheet_name)
			self.c.execute(sql)
			num_rows = worksheet.nrows - 1
			# num_cells = worksheet.ncols - 1
			curr_row = -1
			while curr_row < num_rows:
				curr_row += 1
				data = self.processor.cleanSpecialChar(worksheet.cell_value(curr_row, 0))
				self.c.execute("INSERT INTO %s (Tweet, Score) VALUES (?,?)" % (worksheet_name),
							(data,worksheet.cell_value(curr_row, 1),))    
			self.conn.commit()

	def createTable(self, sql):
		self.c.execute(sql)

	def insertValue(self, sql):
		self.c.execute(sql)
		self.conn.commit()

	# def metadata(table):
	# 	conn = sqlite3.connect(self.database)
	# 	c = conn.cursor()
	# 	c.execute('PRAGMA table_info(%s)' % table)
	# 	data = cur.fetchall()
	# 	for d in data:
	# 		print d[0], d[1], d[3]
