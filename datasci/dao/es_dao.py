# coding:utf8

from elasticsearch import Elasticsearch
from dao.bean.es_conf import ESConf
from constant import VALUE_TYPE_ERROR_TIPS
from elasticsearch import helpers
from dao import Dao


class ESDao(Dao):
    """
      ElasticSearch data access object.
    """

    def __init__(self, conf=None, auto_connect=True):
        """

        Args:
          conf: Configuration for the ES.
          auto_connect:
        """
        super(ESDao, self).__init__(conf)

        self.connector = None
        assert isinstance(conf, ESConf), ValueError(VALUE_TYPE_ERROR_TIPS)

        self._conf = conf

        if auto_connect:
            self.connector = Elasticsearch([conf.host], http_auth=(conf.user, conf.passwd), port=conf.port)

    def connect(self):
        """

        Returns:

        """
        conf = self.conf
        self.connector = Elasticsearch([conf.host], http_auth=(conf.user, conf.passwd), port=conf.port)

    def disconnect(self):
        """

        Returns:

        """
        pass

    def bulk(self, actions):
        """

        Args:
          actions:

        Returns:

        """
        helpers.bulk(self.connector, actions)

    def index(self, index, doc_id, doc_type, body):
        """

        Args:
          index:
          doc_type:
          body:

        Returns:

        """
        self.connector.index(index=index, doc_type=doc_type, id=doc_id, body=body)

    def query(self):
        pass

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, value):
        self._conf = value
