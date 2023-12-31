"""empty message

Revision ID: 6ca2eb05f2b1
Revises: 638337824cf2
Create Date: 2023-06-08 16:35:07.359421

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6ca2eb05f2b1'
down_revision = '638337824cf2'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('labeling',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('datetime', sa.String(length=200), nullable=True),
    sa.Column('tweet_id', sa.String(length=200), nullable=True),
    sa.Column('username', sa.String(length=200), nullable=True),
    sa.Column('text', sa.String(length=2000), nullable=True),
    sa.Column('remove_user', sa.String(length=2000), nullable=True),
    sa.Column('text_cleaning', sa.String(length=2000), nullable=True),
    sa.Column('case_folding', sa.String(length=2000), nullable=True),
    sa.Column('tokenizing', sa.String(length=2000), nullable=True),
    sa.Column('stop_words', sa.String(length=2000), nullable=True),
    sa.Column('stemming', sa.String(length=2000), nullable=True),
    sa.Column('score', sa.Integer(), nullable=True),
    sa.Column('compound', sa.String(length=2000), nullable=True),
    sa.Column('sentimen', sa.String(length=2000), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('preprocess', schema=None) as batch_op:
        batch_op.alter_column('datetime',
               existing_type=sa.DATETIME(),
               type_=sa.String(length=200),
               existing_nullable=True)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('preprocess', schema=None) as batch_op:
        batch_op.alter_column('datetime',
               existing_type=sa.String(length=200),
               type_=sa.DATETIME(),
               existing_nullable=True)

    op.drop_table('labeling')
    # ### end Alembic commands ###
