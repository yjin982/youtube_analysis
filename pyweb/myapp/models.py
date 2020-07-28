from django.db import models

# Create your models here.
# class Board(models.Model):
#     title = models.CharField(max_length=50, blank=True, null=True)
#     content = models.TextField(blank=True, null=True)
#     name = models.CharField(db_column='NAME', max_length=20, blank=True, null=True)  # Field name made lowercase.
#     ip = models.CharField(max_length=20, blank=True, null=True)
#     pw = models.CharField(max_length=20, blank=True, null=True)
#     date = models.DateTimeField(db_column='DATE', blank=True, null=True)  # Field name made lowercase.
#     views = models.IntegerField(default=0, blank=True, null=True,)
#   
#     class Meta:
#         db_table = 'board'
#           
# class BoardComment(models.Model):
#     board = models.ForeignKey(Board, on_delete=models.CASCADE, blank=True, null=True)
#     content = models.TextField(blank=True, null=True)
#     name = models.CharField(db_column='NAME', max_length=20, blank=True, null=True)  # Field name made lowercase.
#     ip = models.CharField(max_length=20, blank=True, null=True)
#     pw = models.CharField(max_length=20, blank=True, null=True)
#     date = models.DateTimeField(db_column='DATE', blank=True, null=True)  # Field name made lowercase.
#   
#     class Meta:
#         db_table = 'board_comment'

class Board(models.Model):
    board_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=50, blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    name = models.CharField(db_column='NAME', max_length=20, blank=True, null=True)  # Field name made lowercase.
    ip = models.CharField(max_length=20, blank=True, null=True)
    pw = models.CharField(max_length=20, blank=True, null=True)
    date = models.DateTimeField(db_column='DATE', blank=True, null=True)  # Field name made lowercase.
    views = models.IntegerField(default=0, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'board'


class BoardComment(models.Model):
    board_comment_id = models.AutoField(primary_key=True)
    board = models.ForeignKey(Board, models.DO_NOTHING, db_column='board', blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    name = models.CharField(db_column='NAME', max_length=20, blank=True, null=True)  # Field name made lowercase.
    ip = models.CharField(max_length=20, blank=True, null=True)
    pw = models.CharField(max_length=20, blank=True, null=True)
    date = models.DateTimeField(db_column='DATE', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'board_comment'